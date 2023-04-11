import os
import glob
import shutil

import numpy as np
from PIL import Image
import cv2

from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import roc_auc_score
from sampling_methods.kcenter_greedy import kCenterGreedy
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import faiss
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import copy

def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)
    return dist


class NN():
    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]


class KNN(NN):
    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        dist = torch.cdist(x, self.train_pts, self.p)
        knn = dist.topk(self.k, largest=False)
        return knn


def prep_dirs(root):
    # make embeddings dir
    embeddings_path = os.path.join('./', 'embeddings', args.category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    return embeddings_path, sample_path, source_code_save_path


def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list


def reshape_mask(masks):
    masks = masks.permute(0, 2, 3, 1)
    masks = (masks.cpu()).numpy()
    b = masks.shape[0]
    masks_resize = np.zeros((b, 28, 28, 3))
    for i in range(masks.shape[0]):
        masks_resize[i] = cv2.resize(masks[i], (28, 28))
    # cv2.imshow('x',masks_resize[0])
    # cv2.waitKey(0)
    return torch.tensor(masks_resize).permute(0, 3, 1, 2)


# imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    def __init__(self, root, root_mask, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.root_mask = root_mask
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        parts = img_path.split('/')
        parts[-5] = self.root_mask
        img_mask_path = "/".join(parts)
        img_mask = Image.open(img_mask_path).convert('RGB')
        img_mask = self.gt_transform(img_mask)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type, img_mask


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


class PatchCore(pl.LightningModule):
    def __init__(self, hparams):
        super(PatchCore, self).__init__()

        self.save_hyperparameters(hparams)

        self.init_features()

        def hook_t(module, input, output):
            self.features.append(output)

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()

        self.data_transforms = transforms.Compose([
            transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.CenterCrop(args.input_size),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((args.load_size, args.load_size)),
            transforms.ToTensor(),
            transforms.CenterCrop(args.input_size)])

        self.inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                                  std=[1 / 0.229, 1 / 0.224, 1 / 0.255])

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm * 255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm * 255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def train_dataloader(self):
        image_datasets = MVTecDataset(root=os.path.join(args.dataset_path, args.category),
                                      root_mask=args.dataset_mask_path,
                                      transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0)
        return train_loader

    def test_dataloader(self):
        test_datasets = MVTecDataset(root=os.path.join(args.dataset_path, args.category),
                                     root_mask=args.dataset_mask_path,
                                     transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0)
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.model.eval()  # to stop running_var move (maybe not critical)
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.embedding_list = []

    def on_test_start(self):
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.index = faiss.read_index(os.path.join(self.embedding_dir_path, 'index.faiss'))
        # if torch.cuda.is_available():
        #     res = faiss.StandardGpuResources()
        #     self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.init_results_list()

    def training_step(self, batch, batch_idx):  # save locally aware patch features
        x, _, _, _, _, x_mask = batch
        x_mask_1 = x_mask[:, 0, :, :]
        index = torch.where(x_mask_1 > 0)
        u = x[index[0], :, index[1], index[2]]
        self.mean_RGB = torch.mean(u, dim=0)
        index_0 = torch.where(x_mask_1 == 0)
        x[index_0[0], :, index_0[1], index_0[2]] = self.mean_RGB
        img_numpy = (x[0].cpu().numpy() + 2) * 40
        # img_numpy=np.uint8(img_numpy.transpose(1, 2, 0))
        # cv2.imshow("x",img_numpy)
        # cv2.waitKey(0)
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        temp = reshape_embedding(np.array(embedding))
        # x_mask = reshape_mask(x_mask)
        # temp_mask = reshape_embedding(np.array(x_mask))
        # temp_mask = np.array(temp_mask).T[0]
        # temp = temp * temp_mask[:, np.newaxis]
        # idx = np.where(temp_mask > 0)
        self.embedding_list.extend(temp)

    def training_epoch_end(self, outputs):
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto',
                                                      eps=0.9)  # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[],
                                             N=int(total_embeddings.shape[0] * args.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]
        self.embedding_coreset = np.float32(self.embedding_coreset)
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        # faiss
        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.index.add(self.embedding_coreset)
        faiss.write_index(self.index, os.path.join(self.embedding_dir_path, 'index.faiss'))

    def test_step(self, batch, batch_idx):  # Nearest Neighbour Search
        x, gt, label, file_name, x_type, x_mask = batch

        x_mask_1 = x_mask[:, 0, :, :]
        index_0 = torch.where(x_mask_1 == 0)
        x[index_0[0], :, index_0[1], index_0[2]] = self.mean_RGB
        # img_numpy = (x[0].cpu().numpy() + 2) * 40
        # img_numpy = np.uint8(img_numpy.transpose(1, 2, 0))
        # cv2.imshow("x", img_numpy)
        # cv2.waitKey(0)
        # extract embedding
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))
        score_patches, _ = self.index.search(embedding_test, k=args.n_neighbors)
        anomaly_map = score_patches[:, 0].reshape((28, 28))
        N_b = score_patches[np.argmax(score_patches[:, 0])]
        w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
        score = w * max(score_patches[:, 0])  # Image-level score
        gt_np = gt.cpu().numpy()[0, 0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
        # anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size)) * (
        #     x_mask[0, 0, :, :].cpu().numpy())
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(file_name)
        # save images
        x = self.inv_normalize(x)
        input_x = cv2.cvtColor(x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        self.save_anomaly_map(anomaly_map_resized_blur, input_x, gt_np * 255, file_name[0], x_type[0])

    def test_epoch_end(self, outputs):
        print("Total pixel-level auc-roc score :")
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        # pred_list_px_lvl_01 = np.array((self.pred_list_px_lvl - min(self.pred_list_px_lvl)) / (
        #             max(self.pred_list_px_lvl) - min(self.pred_list_px_lvl)))
        pred_list_px_lvl_01 = (self.pred_list_px_lvl - min(self.pred_list_px_lvl)) / (
                         max(self.pred_list_px_lvl) - min(self.pred_list_px_lvl))
        Ap_ = average_precision_score(self.gt_list_px_lvl, pred_list_px_lvl_01)
        print("AP:", Ap_)
        f1_max = 0
        n=10
        pred_list_px_lvl_01=np.array(pred_list_px_lvl_01)
        pred_list_px_lvl_01_const=copy.deepcopy(pred_list_px_lvl_01)
        print(pred_list_px_lvl_01)
        for i in range(0, n):
            pred_list_px_index = pred_list_px_lvl_01_const > i / n
            pred_list_px_lvl_01[pred_list_px_index] = 1
            pred_list_px_lvl_01[~pred_list_px_index] = 0
            f1 = f1_score(self.gt_list_px_lvl, pred_list_px_lvl_01)
            if f1 > f1_max:
                f1_max = f1
            print('F1 score:' + str(i / n), f1)
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc, 'AP': Ap_, 'F1 score:': f1_max}
        self.log_dict(values)
        file = open('/home/burly/my_train.txt', 'a')
        file.write(str({self.hparams['category']: values}))
        file.write('\n')
        # 关闭文件
        file.close()
        print('test_epoch_end')


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train', 'test'], default='train')
    parser.add_argument('--dataset_path', default=r'/home/burly/data/MVTec')
    parser.add_argument('--dataset_mask_path', default=r'knn')
    parser.add_argument('--category', default='bottle')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256)
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', default=0.001)
    parser.add_argument('--project_root_path', default=r'./test')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    return args


object_classnames = ['carpet', 'grid', 'leather', 'tile', 'wood']
CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    for i in CLASS_NAMES:
        if i in object_classnames:
            continue
        args.category = i
        trainer = pl.Trainer.from_argparse_args(args,
                                                default_root_dir=os.path.join(args.project_root_path, args.category),
                                                max_epochs=args.num_epochs, gpus=1)
        model = PatchCore(hparams=args)
        if args.phase == 'train':
            trainer.fit(model)
            trainer.test(model)
        elif args.phase == 'test':
            trainer.test(model)
