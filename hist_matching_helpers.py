from skimage import exposure
from skimage.exposure import match_histograms
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
def calc_rmse(orig, matched_imgs, max_dev=0.2):
  #Function that calculates rmse between original and matched images
  #and eliminates matched images with too much change
  #orig: Original image in XYZ color space
  #matched_imgs: List of matched images in RGB
  #max_dev: Maximum allowed deviation from mean
  #returns boolean array of non outlier matched images
  rmse_vals = np.array([np.sqrt(np.mean((matched_imgs[i].astype('float32')-cv2.cvtColor(orig, cv2.COLOR_XYZ2RGB).astype('float32'))**2)) for i in range(0,len(matched_imgs))])
  deviations = (rmse_vals-rmse_vals.mean())/rmse_vals.std()
  not_outlier = deviations < max_dev

  return not_outlier

def choose_best(non_out_mask, pred_diff, em_orig, em_target):
  #Function that chooses best matched image by using model outputs
  #non_out_mask: Mask of non outlier matched images
  #pred_diff: Difference of model prediction after and before matching
  #em_orig: Original emotion
  #em_target: Target emotion
  #Returns the index of the chosen matched image
  pred_diff[~non_out_mask] = -1e6
  pred_diffs_double = pred_diff[:,em_target]-pred_diff[:,em_orig]
  pred_diffs_double[~non_out_mask] = -1e6

  return pred_diffs_double.argsort()[-1:]

def img_blend(im1,im2,alpha):
  #Function that applies image blending
  #im1: Input image 1 in RGB
  #im2: Input image 2 in RGB
  #alpha: Parameter that controls amount of blending
  return (im1*alpha+im2*(1-alpha)).astype('uint8')


image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]

def image_transformation(img_dim, lanczos=True):
    """simple transformation/pre-processing of image data."""

    if lanczos:
        resample_method = Image.LANCZOS
    else:
        resample_method = Image.BILINEAR

    normalize = transforms.Normalize(mean=image_net_mean, std=image_net_std)
    img_transforms = dict()
    img_transforms['train'] = transforms.Compose([transforms.Resize((img_dim, img_dim), resample_method),
                                                  transforms.ToTensor(),
                                                  normalize])

    # Use same transformations as in train (since no data-augmentation is applied in train)
    img_transforms['test'] = img_transforms['train']
    img_transforms['val'] = img_transforms['train']
    img_transforms['rest'] = img_transforms['train']
    return img_transforms

emotions=['anger','disgust','fear','joy','sadness','surprise','neutral']

def classify_img(img, model):
  #input: PIL rgb image
  #output: model presdiction for input image
  img_transform=image_transformation(256)['train']
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  img_inp = torch.unsqueeze(img_transform(img).to(device),0)
  
  out=model(img_inp)
  max_pred = np.argmax(out.detach().cpu().numpy(), 1)    
  #return emotions[int(max_pred)]
  return out.cpu().detach().numpy()[0]



def transform_image(orig, ref_lists, em_orig, em_target, model):
  #Function that does histogram matching
  #orig: Original image in BGR color space
  #em_orig: Original emotion
  #em_target: Target emotion
  source_xyz = cv2.cvtColor(orig, cv2.COLOR_BGR2XYZ)
  ref_list = ref_lists[em_target]
  pred_diff=np.zeros([20,7])
  matched_imgs=[]
  for i in range(0,20):
    matched = match_histograms(source_xyz, cv2.cvtColor(ref_list[i], cv2.COLOR_BGR2XYZ),multichannel=True)
    matched = img_blend(cv2.cvtColor(matched, cv2.COLOR_XYZ2RGB).astype('float32'), cv2.cvtColor(source_xyz, cv2.COLOR_XYZ2RGB).astype('float32'),alpha=0.7 )
    tmp = [cv2.cvtColor(source_xyz, cv2.COLOR_XYZ2RGB),cv2.cvtColor(ref_list[i], cv2.COLOR_BGR2RGB),matched]
    pred = []
    matched_imgs.append(matched)
    for j in range(0,3):
      pred.append(classify_img(Image.fromarray(tmp[j]), model))
    pred_diff[i] = np.array(pred[2])-np.array(pred[0])

  chosen_indexes = choose_best(calc_rmse(source_xyz, matched_imgs),pred_diff, em_orig, em_target)
  matched_imgs_np = np.array(matched_imgs)
  avg=np.mean(matched_imgs_np[chosen_indexes],0).astype('uint8')
  return cv2.cvtColor(avg, cv2.COLOR_RGB2BGR)

