import re
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import f1_score
from deepfakes_dataset import DeepFakesDataset
import os
import cv2
import numpy as np
import torch
from torch import nn, einsum
#from sklearn.metrics import plot_confusion_matrix
from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params,check_frame_every
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from cross_efficient_vit import CrossEfficientViT
from utils import transform_frame
import glob
from os import cpu_count
import json
from multiprocessing.pool import Pool
from progress.bar import Bar
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager
from utils import custom_round, custom_video_round
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate
from transforms.albu import IsotropicResize
import yaml
import argparse
#########################
####### CONSTANTS #######
#########################
MODELS_DIR = "models"
BASE_DIR = '/root/autodl-tmp/base_project/TSFF-Net'
DATA_DIR = "/root/autodl-tmp/base_project/dataset"
TEST_DIR = os.path.join(DATA_DIR, "test_set")
OUTPUT_DIR = os.path.join(MODELS_DIR, "tests")
METADATA_PATH = "/root/autodl-tmp/base_project/data/metadata" # Folder containing all training metadata for DFDC dataset
#########################
####### UTILITIES #######
#########################
def save_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap="Blues")
    threshold = im.norm(confusion_matrix.max())/2.
    textcolors=("black", "white")

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["original", "fake"])
    ax.set_yticklabels(["original", "fake"])
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center",fontsize=12, color=textcolors[int(im.norm(confusion_matrix[i, j]) > threshold)])

    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion.jpg"))
def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')

    fpr, tpr, th = metrics.roc_curve(correct_labels, preds)

    model_auc = auc(fpr, tpr)


    plt.plot(fpr, tpr, label="Model_"+ model_name + ' (area = {:.3f})'.format(model_auc))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(OUTPUT_DIR, model_name +  "_" + opt.dataset + "_acc" + str(accuracy*100) + "_loss"+str(loss)+"_f1"+str(f1)+".jpg"))
    plt.clf()


def read_frames(video_path, videos,config):
    # Get the video label based on dataset selected
    # method = get_method(video_path, DATA_DIR)
    label = 2


    for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        video_folder_name = os.path.basename(video_path)
        video_key = video_folder_name
        if video_key in metadata.keys():
            item = metadata[video_key]
            label = item.get("label", None)
    if label != 2:
    # else:
    #     if "Original" in video_path:
    #         label = 0.
    #     elif "DFDC" in video_path:
    #         val_df = pd.DataFrame(pd.read_csv(VALIDATION_LABELS_PATH))
    #         video_folder_name = os.path.basename(video_path)
    #         video_key = video_folder_name + ".mp4"
    #         label = val_df.loc[val_df['filename'] == video_key]['label'].values[0]
    #     else:
    #         label = 1.

    # Calculate the interval to extract the frames
    # frames_number = len(os.listdir(video_path))

    # if VALIDATION_DIR in video_path:
    #     min_video_frames = int(max(min_video_frames/8, 2))

        frames_paths = os.listdir(video_path)
        frames_paths_dict = {}
        skip_frame = int(config['training']['skip'])
        # min_video_frames = max(int(config['training']['frames-per-video']), 1)
        # Group the faces with the same index, reduce probabiity to skip some faces in the same video
        for path in frames_paths:
            if re.search(r'_(\d+)', path) == None:
                print(path)
                continue
            i = re.search(r'_(\d+)', path).groups(1)
            if i not in frames_paths_dict.keys():
                frames_paths_dict[i] = [path]
            else:
                frames_paths_dict[i].append(path)

        # frames_interval = int(frames_number / len(frames_paths_dict))
        min_video_frames = int(config['training']['frames-per-video'])
        # Select only the frames at a certain interval
        # for key in list(frames_paths_dict.keys()):
        #     random.shuffle(frames_paths_dict[key])
        #     if min_video_frames < len(frames_paths_dict[key]):
        #         if len(frames_paths_dict[key]) < 300:
        #             skip_m = len(frames_paths_dict[key])//min_video_frames
        #             frames_paths_dict[key] = frames_paths_dict[key][::skip_m]
        #             frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]
        #         else:
        #             skip_m = 300//min_video_frames
        #             frames_paths_dict[key] = frames_paths_dict[key][::skip_m]
        #             frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]
        #     else:
        #         del frames_paths_dict[key]
        for key in list(frames_paths_dict.keys()):
            random.shuffle(frames_paths_dict[key])
            skip = len(frames_paths_dict[key])//min_video_frames
            if len(frames_paths_dict[key])>30:        
                frames_paths_dict[key] = frames_paths_dict[key][:30]
                #frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]
            else:
                del frames_paths_dict[key]
        # Select N frames from the collected ones
        video = {}

        for key in frames_paths_dict.keys():

            for index, frame_image in enumerate(frames_paths_dict[key]):
                #transform = create_base_transform(config['model']['image-size'])
                #image = transform(np.asarray(cv2.imread(os.path.join(video_path, frame_image))))
                # image = np.asarray(cv2.cvtColor(cv2.imread(os.path.join(video_path, frame_image)), cv2.COLOR_BGR2GRAY).astype(np.float32))
                image = np.asarray(cv2.imread(os.path.join(video_path, frame_image)))
                if key in video:
                    video[key].append(image)
                else:
                    video[key] = [image]
        videos.append((video_folder_name,video, label))


def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

#########################
#######   MODEL   #######
#########################


# Main body
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_path', default='/root/autodl-tmp/base_project/TSFF-Net/models/efficientnet_checkpoint7_All', type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='Celeb', 
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|DFDC)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', default='/root/autodl-tmp/base_project/TSFF-Net/configs/architecture.yaml',type=str,
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--frames_per_video', type=int, default=50,
                        help="How many equidistant frames for each video (default: 30)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size (default: 32)")
    
    opt = parser.parse_args()
    torch.cuda.empty_cache()
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
 
        
    if os.path.exists(opt.model_path):
        model = CrossEfficientViT(config=config)
        m_model = torch.load(opt.model_path)

        model.load_state_dict(m_model)
        model.eval()
        model = model.cuda()
    else:
        print("No model found.")
        exit()

    #model_name = os.path.basename(opt.model_path)


    #########################
    ####### EXECUTION #######
    #########################


    OUTPUT_DIR = os.path.join(OUTPUT_DIR, opt.dataset)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
   


    NUM_CLASSES = 1
    preds = []

    mgr = Manager()
    paths = []
    videos = mgr.list()
    #videos = shuffle_dataset(videos)

    folders = ["NeuralTextures_c40"]#Face2face_c23  faceshifter_c23
    model_name = folders[0]
    # Read all videos paths
    for folder in folders:
        torch.cuda.empty_cache()
        method_folder = os.path.join(TEST_DIR, folder)
        s = os.listdir(method_folder)
        random.shuffle(s)
        s = s[:]
        for index, video_folder in enumerate(s):
            paths.append(os.path.join(method_folder, video_folder))

    # Read faces
    with Pool(processes=cpu_count()-1) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, videos=videos,config = config),paths):
                pbar.update()
    # video_names = np.asarray([row[2] for row in videos])
    correct_test_labels = np.asarray([row[2] for row in videos])
    #print(correct_test_labels)
    videos_ = [row[1] for row in videos]
    name = [row[0] for row in videos]
    preds = []


    # Perform prediction
    bar = Bar('Predicting', max=len(videos))

    f = open(opt.dataset + "_" + model_name + "_labels.txt", "w+")
    with torch.no_grad():
        for index, video in enumerate(videos_):
            video_faces_preds = []
            scaled_pred = []
            label = correct_test_labels[index]
            f.write(" " + str(name[index]))
            # video_name = video_names[index]
            # f.write(video_name)
            for key in video.keys():
                faces_preds = []
                video_faces = video[key]



                video_faces = DeepFakesDataset([face for face in video_faces], np.asarray([label for i in range(len(video_faces))]), config['model']['image-size'],mode = 'test')
                dl = torch.utils.data.DataLoader(video_faces, batch_size=config['training']['bs'], shuffle=True,
                                                 sampler=None,
                                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                                 pin_memory=False, drop_last=False, timeout=0,
                                                 worker_init_fn=None, prefetch_factor=2,
                                                 persistent_workers=False)
                del video_faces
                for index, (image1, image2, labels) in enumerate(dl):
                    image1 = np.transpose(image1, (0, 3, 1, 2))
                    image2 = np.transpose(image2, (0, 3, 1, 2))
                    image1 = image1.cuda()
                    image2 = image2.cuda()
                    pred = model(image1,image2)
                    scaled_pred = []
                    for idx, p in enumerate(pred):
                        scaled_pred.append(torch.sigmoid(p))
                        #scaled_pred.append(p)
                    faces_preds.extend(scaled_pred)               
                    #y_pred = y_pred.cpu()
                del dl
                #current_faces_pred = sum(faces_preds)/len(faces_preds)
                current_faces_pred = check_frame_every(faces_preds)
                #current_faces_pred = max(faces_preds)
                face_pred = current_faces_pred.cpu().detach().numpy()[0]
                f.write(" " + str(face_pred))
                video_faces_preds.append(face_pred)
            bar.next()
            if len(video_faces_preds) > 1:

                video_pred = [custom_video_round(video_faces_preds)]
            else:

                video_pred = video_faces_preds
            preds.append(video_pred)

            f.write(" --> " + str(video_pred) + "(CORRECT: " + str(label) + ")" +"\n")

        f.close()
        bar.finish()



    #########################
    #######  METRICS  #######
    #########################

    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensor_labels = torch.tensor([[float(label)] for label in correct_test_labels])

    tensor_preds = torch.tensor([[float(pred[0])] for pred in preds])


    loss = loss_fn(tensor_preds, tensor_labels).numpy()

    #accuracy = accuracy_score(np.asarray(preds).round(), correct_test_labels) # Classic way
    accuracy = accuracy_score(custom_round(np.asarray(preds)), correct_test_labels) # Custom way
    f1 = f1_score(correct_test_labels, custom_round(np.asarray(preds)))
    print(model_name, "Test Accuracy:", accuracy, "Loss:", loss, "F1", f1)
    save_roc_curves(correct_test_labels, preds, model_name, accuracy, loss, f1)
    save_confusion_matrix(metrics.confusion_matrix(correct_test_labels,custom_round(np.asarray(preds))))
