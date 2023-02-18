import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

def process_video(path_in, path_out, model, device, count=30):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    print("Opening video: ", path_in)
    path_out += ".mp4"
    video_reader = cv2.VideoCapture(path_in)
    fps = video_reader.get((cv2.CAP_PROP_FPS))
    width  = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS: {} Widht: {} Height {}".format(fps, width, height))

    video_writer = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    success = True
    while success:
        print("Process frame: {}/{}".format(frame_count + 1, count))
        success, image = video_reader.read()
        frame_count += 1
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            image_tensor = torch.clamp(model(image_tensor.to(device)), -1.0, 1.0)
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = np.rint(((np.transpose(image_numpy, (1, 2, 0)) + 1.0) / 2.0) * 255.0).astype(np.uint8)
        video_writer.write(image_numpy)

        if frame_count == count:
            print("Succesfully processed video: ", path_out)
            break
    if frame_count != count:
        print("Failed to process frames")
