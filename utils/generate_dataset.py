import os
import shutil
import numpy as np

def move(folder_A,folder_B):
    for root, dirs, files in os.walk(folder_A):
        for file in files:
            source_file_path = os.path.join(root, file)
            target_file_path = os.path.join(folder_B, file)
            shutil.copy(source_file_path, target_file_path)

def move_seq(folder_A,folder_B):
    for root, dirs, files in os.walk(folder_A):
        for file in files:
            if not (file.endswith('001.png') or file.endswith("002.png")):
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(folder_B, file)
                shutil.copy(source_file_path, target_file_path)

def mkdir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    else:
        print(f'{dir} exists')


def generate_key_frame(target, force_dir, output_folder):
    for force_name in os.listdir(force_dir):
        clipname = os.path.splitext(force_name)[0]
        force_file  = os.path.join(force_dir,force_name)
        force_z = np.genfromtxt(force_file,dtype=float,usecols=2)
        max_index = np.argmin(force_z)+1
        img_name = f'{clipname}_{str(max_index).zfill(3)}.png'
        img_name_max = f'{clipname}_max.png'

        
        # print("max:",os.path.join(root,'Videos',img_name))
        if os.path.exists(os.path.join(target,'Videos','train',img_name)):
            shutil.copyfile(os.path.join(target,'Videos','train',img_name), os.path.join(output_folder,img_name_max)) 
        elif os.path.exists(os.path.join(target,'Videos','valid',img_name)):
            shutil.copyfile(os.path.join(target,'Videos','valid',img_name), os.path.join(output_folder,img_name_max)) 
        else:
            print(f'{img_name} doesn\'t exsist')

            
        min_index = np.argmax(force_z)+1
        img_name = f'{clipname}_{str(min_index).zfill(3)}.png'
        img_name_min = f'{clipname}_min.png'
        # print("min:",os.path.join('Videos',img_name))

        if os.path.exists(os.path.join(target,'Videos','train',img_name)):
            shutil.copyfile(os.path.join(target,'Videos','train',img_name), os.path.join(output_folder,img_name_min)) 
        elif os.path.exists(os.path.join(target,'Videos','valid',img_name)):
            shutil.copyfile(os.path.join(target,'Videos','valid',img_name), os.path.join(output_folder,img_name_min)) 
        else:
            print(f'{img_name} doesn\'t exsist')

if __name__ == "__main__":
    """folder of downloaded Mus-V"""
    root = 'Multimodal Ultrasound Vascular Segmentation'
    """target data folder"""
    # target = 'data/Multimodal Ultrasound Vascular Segmentation'
    target = 'data/test'
    mkdir(target)
    mkdir(os.path.join(target, 'Forces'))
    mkdir(os.path.join(target, 'Forces_img'))
    mkdir(os.path.join(target, "Annotations"))
    mkdir(os.path.join(target, r"Videos"))
    mkdir(os.path.join(target, r"Videos/valid"))
    mkdir(os.path.join(target, r"Videos/train"))
    mkdir(os.path.join(target, r"Videos_seq"))
    mkdir(os.path.join(target, r"Videos_seq/train"))
    mkdir(os.path.join(target, r"Videos_seq/valid"))
    shutil.copytree(os.path.join(root,'Forces'), os.path.join(target,'Forces'),dirs_exist_ok=True)
    print('Forces Finished')
    move(os.path.join(root,"Annotations"), os.path.join(target,"Annotations"))
    print('Annotations Finished')
    move(os.path.join(root,r"Videos/valid"), os.path.join(target,r"Videos/valid"))
    move(os.path.join(root,r"Videos/train"), os.path.join(target,r"Videos/train"))
    move_seq(os.path.join(root,r"Videos/valid"), os.path.join(target,r"Videos_seq/valid"))
    move_seq(os.path.join(root,r"Videos/train"), os.path.join(target,r"Videos_seq/train"))
    print('Videos Finished')
    generate_key_frame(target, os.path.join(root,"Forces"),os.path.join(target, 'Forces_img'))

