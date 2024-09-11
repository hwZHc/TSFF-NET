import re
import time
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, choice
from vit_pytorch import ViT
import numpy as np
import os
import random
import json
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from cross_efficient_vit import CrossEfficientViT
import uuid
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
import cv2
from transforms.albu import IsotropicResize
import glob
import pandas as pd
from tqdm import tqdm
from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params,check_correct0
from sklearn.utils.class_weight import compute_class_weight 
from torch.optim import lr_scheduler
import collections
from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse
from multiprocessing import Manager
from PIL import Image

BASE_DIR = '/root/autodl-tmp/base_project'
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
MODELS_PATH = "models"
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata") # Folder containing all training metadata for DFDC dataset
VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels.csv")

def find_top_n_frames_with_max_average_difference(image_sequence, n=30):
    # 将图片序列转换为 NumPy 数组
    frames_array = np.asarray(image_sequence)
    
    # 计算相邻帧之间的差值
    frame_diff = np.diff(frames_array,axis=0)

    # 计算差值的绝对值
    frame_diff_abs = np.abs(frame_diff)

    # 计算每个差值张量的平均值
    average_diff = [np.mean(frame_diff_abs_) for frame_diff_abs_ in frame_diff_abs]

    # 获取平均差值最大的前n个帧的索引
    top_n_indices = np.argsort(average_diff)[-n:]

    # 获取平均差值最大的前n个帧
    top_n_frames = []
    for i in top_n_indices:
        top_n_frames.append(frames_array[i])


    return top_n_frames



def read_frames(video_path, train_dataset, validation_dataset, config):

    # Get the video label based on dataset selected
    #method = get_method(video_path, DATA_DIR)
    if "training_set" in video_path or "validation_set" in video_path:
        for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
            with open(json_path, "r") as f:
                metadata = json.load(f)
            video_folder_name = os.path.basename(video_path)
            video_key = video_folder_name
            if video_key in metadata.keys():
                item = metadata[video_key]
                label = item.get("label", None)
                if label == 0:
                    frame_pic = int(config['training']['real-pic'])
                else:
                    frame_pic = int(config['training']['fake-pic'])
            
        if label == None:
            print("NOT FOUND", video_path)

    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    
    # if VALIDATION_DIR in video_path:

    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}
    skip_frame = int(config['training']['skip'])

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        if re.search(r'_(\d+)', path) == None:
            continue
            #print(video_path,path)
        i = re.search(r'_(\d+)', path).groups(1)
        if i not in frames_paths_dict.keys():
            frames_paths_dict[i] = [path]
        else:
            frames_paths_dict[i].append(path)
    video = {}
        
    
    # Select only the frames at a certain interval
    for key in list(frames_paths_dict.keys()):
        if len(frames_paths_dict[key]) > 20:
            frames_paths_dict[key] = sorted(frames_paths_dict[key], key=lambda x: int(x.split('_')[0]))
            frames_paths_dict[key] = frames_paths_dict[key][2:-2]
            if len(frames_paths_dict[key]) < 60:
                m = frames_paths_dict[key][-1]
                for i in range(40-len(frames_paths_dict[key])):

                    frames_paths_dict[key].append(m)
            random.shuffle(frames_paths_dict[key])


            for key in frames_paths_dict.keys():

                for index, frame_image in enumerate(frames_paths_dict[key]):
                    image_path = os.path.join(video_path, frame_image)
                    try:
                        image = cv2.imread(image_path)
                        if image is None:
                            print(image_path)
                    except Exception as e:
                        print(image_path)                       
                    image = np.asarray(image)

                    if key in video:
                        video[key].append(image)
                    else:
                        video[key] = [image]
            for key in video.keys():
                #v = find_top_n_frames_with_max_average_difference(video[key])
                v = video[key][:frame_pic]
                if TRAINING_DIR in video_path:           
                    train_dataset.append((v, label))
                else:
                    validation_dataset.append((v, label))


# Main body
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=20, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='All', 
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, default='configs/architecture.yaml',
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=10, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    
    opt = parser.parse_args()

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    model = CrossEfficientViT(config=config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    #optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'], last_epoch=-1)
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1
    else:
        print("No checkpoint loaded.")


    print("Model Parameters:", get_n_params(model))
   
    #READ DATASET
    if opt.dataset != "All":
        folders = ["Original", opt.dataset]
    else:
        folders = ["NeuralTextures_c23"]
        # folders = ["Original", "DFDC", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

    sets = [TRAINING_DIR, VALIDATION_DIR]

    
    paths = []
    for dataset in sets:
        for folder in folders:
            subfolder = os.path.join(dataset, folder)
            s = os.listdir(subfolder)
            # random.shuffle(s,random=r)
            random.shuffle(s)
            if dataset == sets[0]:
                #s = s[:10]
                s = s[:]
            else:
                s = s[:]          
            for index, video_folder_name in enumerate(s):
                if index == opt.max_videos:
                    break
                if os.path.isdir(os.path.join(subfolder, video_folder_name)):
                    paths.append(os.path.join(subfolder, video_folder_name))
                

    mgr = Manager()
    train_dataset = mgr.list()
    validation_dataset = mgr.list()

    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, train_dataset=train_dataset, validation_dataset=validation_dataset, config=config),paths):
                pbar.update()
    # train_dataset1 = [(subset, label) for subset, label in train_dataset]
    # validation_dataset1 = [(subset, label) for subset, label in validation_dataset]
    
        
        
    train_samples = len(train_dataset)
    #train_dataset1 = [(sub, subset[1]) for subset in train_dataset for sub in subset[0]]
    train_dataset1 = [(sub[i], val) for sub, val in train_dataset for i in range(len(sub))]
    
    train_dataset1 = shuffle_dataset(train_dataset1)
    validation_samples = len(validation_dataset)
    #validation_dataset1 = [(sub, subset[1]) for subset in validation_dataset for sub in subset[0]]
    validation_dataset1 = [(sub[i], val) for sub, val in validation_dataset for i in range(len(sub))]
    validation_dataset1 = shuffle_dataset(validation_dataset1)
    
    

    # Print some useful statistics
    print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(image[1] for image in train_dataset)
    print(train_counters)
    
    fake_pic = int(config['training']['fake-pic'])
    real_pic = int(config['training']['real-pic'])
    
    class_weights = train_counters[0]*real_pic / (train_counters[1]*fake_pic)
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(image[1] for image in validation_dataset)
    print(val_counters)
    print("___________________")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))
    #loss_fn = TripletLoss(margin=0.2)

    # Create the data loaders
    validation_labels = np.asarray([row[1] for row in validation_dataset1])
    labels = np.asarray([row[1] for row in train_dataset1])


    train_dataset = DeepFakesDataset([row[0] for row in train_dataset1], labels, config['model']['image-size'])
    del train_dataset1
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=True, drop_last=True, timeout=0,
                                 worker_init_fn=None, prefetch_factor=16,
                                 persistent_workers=True)
    del train_dataset
    

    validation_dataset = DeepFakesDataset([row[0] for row in validation_dataset1], validation_labels, config['model']['image-size'], mode='val')
    del validation_dataset1
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=True, drop_last=True, timeout=0,
                                    worker_init_fn=None, prefetch_factor=16,
                                    persistent_workers=True)
    del validation_dataset
    
    
       
    model = model.cuda()
    
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, opt.num_epochs + 1):
        
        model.train()
        if not_improved_loss == opt.patience:
            break
        counter = 0
        total_loss = 0
        total_val_loss = 0
        
        #bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*config['training']['bs'])+len(val_dl))
        bar = ChargingBar('EPOCH #' + str(t), max=len(dl)+len(val_dl))
        train_correct = 0
        positive = 0
        negative = 0

        epoch_start_time = time.time()
        
        for index, (image1,image2, labels) in enumerate(dl):

            image1 = np.transpose(image1, (0, 3,1, 2))
            image2 = np.transpose(image2, (0, 3,1, 2))

            labels = labels.unsqueeze(1)
            image1 = image1.cuda()
            image2 = image2.cuda()

            y_pred = model(image1,image2)

            y_pred = y_pred.cpu()
            loss = loss_fn(y_pred, labels.float())

            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            
            #loss.requires_grad_
                                   
            optimizer.zero_grad()
            loss.backward()            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()            
            counter += 1
            total_loss += round(loss.item(), 2)
            #for i in range(config['training']['bs']):
            bar.next()

        epoch_end_time = time.time()
        print(f"Epoch {t} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")

        train_acc = train_correct/(counter*config['training']['bs'])
        print("\nLoss: ", total_loss/counter, "Accuracy: ",train_acc ,"Train 0s: ", negative, "Train 1s:", positive)
        

        val_counter = 0
        val_correct = 0
        val_positive = 0
        val_negative = 0

       
        train_correct /= train_samples
        total_loss /= counter
        model.eval()
        with torch.no_grad():
            for index, (val_image1,val_image2, val_labels) in enumerate(val_dl):

                val_image1 = np.transpose(val_image1, (0, 3, 1, 2))
                val_image2 = np.transpose(val_image2, (0, 3, 1, 2))

                val_image1 = val_image1.cuda()
                val_image2 = val_image2.cuda()
                val_labels = val_labels.unsqueeze(1)
                val_pred = model(val_image1,val_image2)
                val_pred = val_pred.cpu()
                val_loss = loss_fn(val_pred, val_labels.float())
                total_val_loss += round(val_loss.item(), 2)

                corrects, positive_class, negative_class = check_correct(val_pred, val_labels)

                val_correct += corrects
                val_positive += positive_class
                val_negative += negative_class
                val_counter += 1            
                bar.next()
        
        scheduler.step() 
        bar.finish()
        

        total_val_loss /= val_counter
        
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0
        
        previous_loss = total_val_loss
        val_acc = val_correct/(val_counter*config['training']['bs'])
        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" + str(total_loss) + " accuracy:" + str(train_acc) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_acc) + " val_0s:" + str(val_negative) + "/" + str(np.count_nonzero(validation_labels == 0)) + " val_1s:" + str(val_positive) + "/" + str(np.count_nonzero(validation_labels == 1)))
    
        
        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)
        torch.save(model.state_dict(), os.path.join(MODELS_PATH,  "efficientnet_checkpoint" + str(t) + "_" + opt.dataset))
        torch.cuda.empty_cache()

         

        
        
