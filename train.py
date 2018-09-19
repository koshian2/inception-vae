import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

import datetime, pickle, os, zipfile

import google.colab
import googleapiclient.discovery
import googleapiclient.http

class GoogleColabUtils:
    # Reference : https://qiita.com/kakinaguru_zo/items/33dbe24276915124f545
    def __init__(self):
        google.colab.auth.authenticate_user()
        self.drive_service = googleapiclient.discovery.build('drive', 'v3')

    def load_file(self, filename):
        file_list = self.drive_service.files().list(q="name='" + filename + "'").execute().get('files')

        # Get file-id
        file_id = None
        for file in file_list:
            if file.get('name') == filename:
                file_id = file.get('id')
                break

        if file_id is None:
            # if filename not found...
            print(filename + "is not found.")
        else:
            # upload to colab environment
            with open(filename, 'wb') as f:
                request = self.drive_service.files().get_media(fileId=file_id)
                media = googleapiclient.http.MediaIoBaseDownload(f, request)

                done = False
                while not done:
                    progress_status, done = media.next_chunk()
                    print(100*progress_status.progress(), end="")
                    print("% finished")

        print('Importing from GoogleDrive to the Colab environment is finished.')

    def save_to_googledrive(self, dataset):
        saving_filename = dataset+".zip"

        file_metadata = {
          'name': saving_filename,
          'mimeType': 'application/octet-stream'
        }
        media = googleapiclient.http.MediaFileUpload(saving_filename, 
                                mimetype='application/octet-stream',
                                resumable=True)
        created = self.drive_service.files().create(body=file_metadata,
                                               media_body=media,
                                               fields='id').execute()

    def extract_data(self):
        if not os.path.exists("jaffe-data.zip"):
            self.load_file("jaffe-data.zip")
        if not os.path.exists("jaffe-data"):
            os.mkdir("jaffe-data")
        with zipfile.ZipFile("jaffe-data.zip", "r") as zip:
            zip.extractall("jaffe-data")

google_colab = GoogleColabUtils()
google_colab.load_file("jaffe_vae_model.py")
google_colab.extract_data()

from jaffe_vae_model import VAE

def train(vae, loader, optimizer, history, epoch):
    vae.train()
    print(f"\nEpoch: {epoch+1:d} {datetime.datetime.now()}")
    train_loss = 0
    samples_cnt = 0
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs = inputs.to(vae.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(inputs)

        loss = vae.loss_function(recon_batch, inputs, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        samples_cnt += inputs.size(0)

        print(batch_idx, len(loader), f"Loss: {train_loss/samples_cnt:f}")

        if batch_idx == 0:
            save_image(recon_batch[:16], f"{vae.model_name}/reconstruction_epoch{epoch}.png", nrow=4)

    save_image(vae.sampling(), f"{vae.model_name}/sampling_epoch{epoch}.png", nrow=4)
    history["loss"].append(train_loss/samples_cnt)

# save results
def save_history(modelname, history):
    with open(f"{modelname}/{modelname}_history.dat", "wb") as fp:
        pickle.dump(history, fp)

def save_to_zip(modelname):
    with zipfile.ZipFile(f"{modelname}.zip", "w") as zip:
        for file in os.listdir(modelname):
            zip.write(f"{modelname}/{file}", file)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model
    net = VAE(True, 1, device)

    # init
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    if device == "cuda":
        net = net.cuda()
        torch.backends.cudnn.benchmark=True
    net.to(device)

    # data
    data_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    jaffe_datasets = datasets.ImageFolder(root="jaffe-data", transform=data_transform)
    loader = torch.utils.data.DataLoader(jaffe_datasets, batch_size=32, shuffle=True)
    # history
    history = {"loss":[]}

    # create output directory
    if not os.path.exists(net.model_name):
        os.mkdir(net.model_name)

    #train
    for i in range(200):
        train(net, loader, optimizer, history, i)
    
    # save results
    save_history(net.model_name , history)
    save_to_zip(net.model_name)

    # write to google drive
    google_colab.save_to_googledrive(net.model_name)


if __name__ == "__main__":
    main()
