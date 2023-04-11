import os.path
import shutil
root=r'/home/burly/Downloads_ubuntu/project/patchcore-inspection-mainauc92/src/log/cl_wideresnet50_layer2'
des_root='/home/burly/data/MVTec_mask'
class_names=os.listdir(root)
for class_name_index in range(len(class_names)):
    classes_root_path=os.path.join(root,class_names[class_name_index])
    classes_root_dirs=os.listdir(classes_root_path)
    for classes_root_dir_index in range(len(classes_root_dirs)):
        test_train_path=os.path.join(classes_root_path,classes_root_dirs[classes_root_dir_index])
        broken=os.listdir(test_train_path)
        for broken_index in range(len(broken)):
            broken_path=os.path.join(test_train_path,broken[broken_index])
            parts = broken_path.split('/')
            imgs_paths=os.listdir(broken_path)
            for img in imgs_paths:
                if img[:4]=='mask':
                    dst_path=os.path.join(des_root,parts[-3],parts[-2],parts[-1])
                    os.makedirs(dst_path,exist_ok=True)
                    shutil.copy(os.path.join(broken_path,img), dst_path)


