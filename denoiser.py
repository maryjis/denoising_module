from datasets import DenoisingDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from unet import UNet
from torch.autograd import Variable
import os
import pandas as pd
from PIL import Image
import torchvision.models as models
import torchvision
import argparse

def augmentation_simple(image_size):
    return torchvision.transforms.Compose([
        torchvision.transforms.Lambda(preprocess_numpy),
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def preprocess_numpy(numpy_array):
    numpy_array = (numpy_array - numpy_array.min()) / (numpy_array.max() - numpy_array.min()) * 255
    img = Image.fromarray(numpy_array.astype(np.uint8))
    img = img.convert("RGB")
    return img


def padding_left(numpy_image, frame_size):
    result_image = np.zeros((frame_size, numpy_image.shape[1])).astype(np.float16)
    result_image[:numpy_image.shape[0], :] = numpy_image
    return result_image


def cut_image_into_frames(numpy_image, frame_size):
    if (numpy_image.shape[0] < frame_size):
        result_image = padding_left(numpy_image, frame_size)
        return [result_image]
    elif (numpy_image.shape[0] == frame_size):
        return [numpy_image]
    else:
        results = []
        frame_number = int(np.ceil(numpy_image.shape[0] / frame_size))
        for frame_id in range(0, frame_number):
            if (numpy_image.shape[0] >= (frame_id + 1) * frame_size):
                result_image = numpy_image[frame_id * frame_size:(frame_id + 1) * frame_size, :]
                results.append(result_image)
            else:
                result_image = padding_left(numpy_image[frame_id * frame_size:, :], frame_size)
                results.append(result_image)
        return results


def combine_frames(results, size):
    result_frame = np.concatenate(results, axis=0)
    return result_frame[:size, :]


def load_models(device):
    model_denoiser = UNet(n_classes=1,depth=5, padding=True,batch_norm=True)
    model_denoiser.load_state_dict(torch.load("data/denoising_autoencoder.pth", map_location=torch.device(device)))
    model_denoiser = model_denoiser.to(device)

    model_classifier = models.resnet18(pretrained=True)
    num_ftrs = model_classifier.fc.in_features
    model_classifier.fc = torch.nn.Linear(num_ftrs, 1)
    model_classifier.load_state_dict(torch.load("data/classifier_model.pth", map_location=torch.device(device)))
    model_classifier = model_classifier.to(device)

    model_classifier.eval()
    model_denoiser.eval()
    return model_denoiser, model_classifier


def denoise_data(test_path, result_path, device_name):
    print("----Start denoising process ---------")
    device = torch.device(device_name)
    test_dataset = DenoisingDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_denoiser, model_classifier = load_models(device)
    augmentation = augmentation_simple((756, 80))

    denoised_path = os.path.join(result_path, "denoised")
    if (not os.path.exists(denoised_path)):
        os.mkdir(denoised_path)

    results_data = []
    with torch.no_grad():
        for path in test_loader:
            mel = np.load(path[0])
            augmented_mel = augmentation(mel)
            augmented_mel =augmented_mel[np.newaxis, :, :, :].to(device)
            prediction = model_classifier(augmented_mel)
            if (prediction[0] > 0.5):
                try:
                    mel_frames = cut_image_into_frames(mel, 80)
                    output_clean_frames = []

                    for mel_noise_frame in mel_frames:
                        mel_noise_frame = torch.tensor(mel_noise_frame, dtype=torch.float)
                        mel_noise_frame = mel_noise_frame[np.newaxis, np.newaxis, :, :]

                        img_noisy = Variable(mel_noise_frame).to(device)

                        output = model_denoiser(img_noisy)
                        output_clean_frames.append(output[0, 0, :, :].cpu().detach().numpy())

                    output_clean = combine_frames(output_clean_frames, mel.shape[0])
                    result_packeges = path[0].split("/")[-2:]

                    user_package = os.path.join(denoised_path, result_packeges[0])
                    if (not os.path.exists(user_package)):
                        os.mkdir(user_package)
                    result_numpy_path = os.path.join(user_package, result_packeges[1])

                    np.save(result_numpy_path, output_clean)
                    results_data.append({"file_name": path[0],
                                         "result": "noisy",
                                         "denoised_file": result_numpy_path})
                except Exception as e:
                    print("Can not denoise file: " + path[0])
                    print(e)


            else:
                results_data.append({"file_name": path[0],
                                     "result": "clean",
                                     "denoised_file": ""})

    result = pd.DataFrame.from_dict(results_data)
    result.to_csv(os.path.join(result_path, "results.csv"), index=False)
    print("----End denoising process ---------")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_dataset", help="specify path to datset")
    parser.add_argument("path_to_results", help="specify path to result")
    parser.add_argument("--device_name", help="increase output verbosity", default="cuda:0")
    args = parser.parse_args()

    denoise_data(args.path_to_dataset, args.path_to_results, args.device_name)
