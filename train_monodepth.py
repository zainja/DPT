import copy
import cv2
import time
import torch
import os
from torchvision.transforms import Compose
import torch.nn.functional as F
from dpt.models import DPTDepthModel
from dataset.rhd import RHD
from numpy.core.numeric import Inf
from dpt.loss import ScaleAndShiftInvariantLoss
import torch.utils.data as data
# Load Data
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

DATA_PATH = ""
ANNO_PATH = os.path.join(DATA_PATH, "anno_training.pickle")
MODEL_PATH = "weights/dpt_large-midas-2f21e586.pt"

net_w = net_h = 256


def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    since = time.time()
    device = torch.device("cuda")
    model.to(device)
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = Inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate over data.
        for i, batch in enumerate(dl):
            for k, v in batch.items():
                batch[k] = v.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                prediction = model(batch["image"])
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=(net_w, net_h),
                    mode="bilinear",
                    align_corners=False,
                )
                loss = criterion(prediction, batch["depth"])

                # backward + optimize only if in training phase

                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * batch["image"].size(0)

        epoch_loss = running_loss / len(dataloader.dataset)

        print('Loss: {:.4f}'.format(epoch_loss))

        # deep copy the model
        if epoch_loss > best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


model = DPTDepthModel(
    path=MODEL_PATH,
    backbone="vitl16_384",
    non_negative=True,
    enable_attention_hooks=False,
)
transform = Compose(
    [
        Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)
loss = ScaleAndShiftInvariantLoss()
dataset = RHD(DATA_PATH, ANNO_PATH, transform)
dl = data.DataLoader(
    dataset, batch_size=4, num_workers=0, shuffle=False, pin_memory=True
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_model(model, dl, loss, optimizer)
