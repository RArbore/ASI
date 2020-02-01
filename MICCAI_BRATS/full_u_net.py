import torch
from torchvision import transforms
import time
import math
import os


#Define constants.

DATA_SIZE = 149*173*192

DATA_DIMENSIONS = [149, 173, 192]

BATCH_SIZE = 5

TRAIN_DATA_SIZE = 300

VALIDATION_DATA_SIZE = 20

TESTING_DATA_SIZE = 15

NUM_EPOCHS = 500

BCE_COEFFICIENT = 14

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

folder = ""


#Define the network.

class UNet(torch.nn.Module):
    
    def __init__(self):
        super(UNet, self).__init__()
        self.seg1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 10, (3, 3, 6), (2, 2, 2)),
            torch.nn.ReLU()
        )
        self.seg2 = torch.nn.Sequential(
            torch.nn.MaxPool3d((2, 2, 2)),
            torch.nn.Conv3d(10, 25, (3, 5, 5), (2, 2, 2)),
            torch.nn.ReLU()
        )
        self.seg3 = torch.nn.Sequential(
            torch.nn.MaxPool3d((2, 2, 2)),
            torch.nn.Conv3d(25, 60, (5, 5, 5)),
            torch.nn.ReLU()
        )
        self.seg4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(60, 25, (5, 5, 5)),
            torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners = False),
            torch.nn.ReLU()
        )
        self.seg5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(50, 10, (3, 5, 5), (2, 2, 2)),
            torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners = False),
            torch.nn.ReLU()
        )
        self.seg6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(20, 1, (3, 3, 6), (2, 2, 2))
        )
        self.sigmoid = torch.nn.Sequential(
            torch.nn.Sigmoid()
        )
        self.dropout = torch.nn.Sequential(
            torch.nn.Dropout(p=0.4)
        )
    
    def forward(self, input):
        before = input.view(input.size(0), 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2]).float()/256.0
        
        s1 = self.dropout(self.seg1(before))
        s2 = self.dropout(self.seg2(s1))
        s3 = self.dropout(self.seg3(s2))
        s4 = torch.cat((self.seg4(s3), s2), dim=1)
        s5 = torch.cat((self.seg5(s4), s1), dim=1)
        s6 = self.seg6(s5)
        
        out = s6.view(input.size(0), DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2])  
        out= self.sigmoid(out)
        return out*torch.tensor(0.998)+torch.tensor(0.001)
    
    
#Define a function for saving images.
    
def save_image(tensor, filename):
    ndarr = tensor.mul(255).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)


#Define pixel-wise binary cross entropy.
        
def pixel_BCE(output_tensor, label_tensor):
    label_tensor = torch.min(label_tensor, torch.ones(label_tensor.size()).float().to(device))
    
    loss_tensor = BCE_COEFFICIENT*label_tensor*torch.log(output_tensor)+(torch.ones(label_tensor.size()).to(device)-label_tensor)*torch.log(torch.ones(output_tensor.size()).to(device)-output_tensor)
    
    return torch.mean(loss_tensor)*torch.tensor(-1).to(device)


#Define error metric.
def error_metric(output_tensor, label_tensor):
    rounded = torch.round(output_tensor).float().to(device)
    diff = (rounded-label_tensor).view(-1).float().to(device)
    label = label_tensor.view(-1).float().to(device)
    tumor_size = torch.sum(label).float().to(device)
    false_positives = torch.sum(torch.max(torch.zeros(diff.size()).float().to(device), diff)).to(device)
    false_negatives = -1*torch.sum(torch.min(torch.zeros(diff.size()).float().to(device), diff)).to(device)
    return false_positives/tumor_size, false_negatives/tumor_size, (false_positives+false_negatives)/tumor_size
        
    
#Declare and train the network.
    
def train_model(data):
    model = UNet().to(device)
    
    opt = torch.optim.Adadelta(model.parameters())
        
    current_milli_time = lambda: int(round(time.time() * 1000))
    
    before_time = current_milli_time()

    print("Beginning Training.")
    print("")

    if not os.path.isdir(folder+"/control_u_net_image_output"):
        os.mkdir(folder+"/control_u_net_image_output")
    if not os.path.isdir(folder+"/during_training_models"):
        os.mkdir(folder+"/during_training_models")
        
    f = open(folder+"/during_training_performance.txt", "a")
    
    for epoch in range(0, NUM_EPOCHS):
        batch_loss = 0 
        e_f_pos = 0 
        e_f_neg = 0
        e_error_m = 0
        epoch_before_time = current_milli_time()
        for batch in (range(0, int(TRAIN_DATA_SIZE/BATCH_SIZE))):
            input_tensor = data[0][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].to(device)
            label_tensor = data[1][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].to(device)
            label_tensor = torch.min(torch.ceil(label_tensor.float()), torch.ones(label_tensor.size()).float().to(device)).to(device)
            opt.zero_grad()
            output_tensor = model(input_tensor.float())
            if batch == 1:
                if not os.path.isdir(folder+"/control_u_net_image_output/epoch_"+str(epoch)):
                    os.mkdir(folder+"/control_u_net_image_output/epoch_"+str(epoch))
                for i in range(0, DATA_DIMENSIONS[0]):
                    save_image(output_tensor[0, i, :, :], folder+"/control_u_net_image_output/epoch_"+str(epoch)+"/"+str(i+1)+".png")
            train_loss = pixel_BCE(output_tensor, label_tensor.float())
            train_loss.backward()
            opt.step()
            f_pos, f_neg, error_m = error_metric(output_tensor, label_tensor.float())
            e_f_pos += f_pos.item()
            e_f_neg += f_neg.item()
            e_error_m += error_m.item()
            train_loss_item = train_loss.item()
            batch_loss += train_loss_item
            #print("Batch "+str(batch+1)+" Loss : "+str(train_loss_item)+" Took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")
        valid_loss = 0
        for i in range(0, int(VALIDATION_DATA_SIZE/BATCH_SIZE)):
            test_data = data[0:2, TRAIN_DATA_SIZE+i*BATCH_SIZE:TRAIN_DATA_SIZE+(i+1)*BATCH_SIZE, :, :, :]
            input_tensor = test_data[0].to(device)
            label_tensor = test_data[1].to(device)
            label_tensor = torch.min(torch.ceil(label_tensor.float()), torch.ones(label_tensor.size()).float().to(device)).to(device)
            output_tensor = model(input_tensor.float())
            loss = pixel_BCE(output_tensor, label_tensor.float())
            valid_loss += loss.item()
        valid_loss /= VALIDATION_DATA_SIZE/BATCH_SIZE
        epoch_loss = batch_loss/(TRAIN_DATA_SIZE/BATCH_SIZE)
        false_positives = e_f_pos/(TRAIN_DATA_SIZE/BATCH_SIZE)
        false_negatives = e_f_neg/(TRAIN_DATA_SIZE/BATCH_SIZE)
        epoch_error_metric = e_error_m/(TRAIN_DATA_SIZE/BATCH_SIZE)
        epoch_after_time = current_milli_time()
        seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
        minutes = math.floor(seconds / 60)
        seconds = seconds % 60
        print("Epoch "+str(epoch+1)+" Loss : "+str(epoch_loss)+"   Validation Loss : "+str(valid_loss)+"   Error Metrics :    F_P : "+str(false_positives)+"   F_N : "+str(false_negatives)+"   Error Metric : "+str(epoch_error_metric)+" Took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")
        f.write(str(epoch+1)+" "+str(epoch_loss)+" "+str(valid_loss)+" "+str(false_positives)+" "+str(false_negatives)+" "+str(epoch_error_metric)+"\n")
        torch.save(model.state_dict(), folder+"/during_training_models/model_at_e"+str(epoch+1)+".pt")
        if epoch+1 == NUM_EPOCHS:
            f.write("\n")

    f.close()
        
    after_time = current_milli_time()
    
    torch.save(model.state_dict(), folder+"/model.pt")
    
    t_test_loss = 0
    t_f_pos = 0
    t_f_neg = 0
    t_error_m = 0
    
    indiv_losses = torch.zeros([int(TESTING_DATA_SIZE/BATCH_SIZE)])
    indiv_errors = torch.zeros([int(TESTING_DATA_SIZE/BATCH_SIZE)])
    
    for i in range(0, int(TESTING_DATA_SIZE/BATCH_SIZE)):
        test_data = data[0:2, TRAIN_DATA_SIZE+VALIDATION_DATA_SIZE+i*BATCH_SIZE:TRAIN_DATA_SIZE+VALIDATION_DATA_SIZE+(i+1)*BATCH_SIZE, :, :, :]
        input_tensor = test_data[0].to(device)
        label_tensor = test_data[1].to(device)
        label_tensor = torch.min(torch.ceil(label_tensor.float()), torch.ones(label_tensor.size()).to(device)).to(device)
        output_tensor = model(input_tensor.float())
        test_loss = pixel_BCE(output_tensor, label_tensor.float())
        f_pos, f_neg, error_m = error_metric(output_tensor, label_tensor.float())
        indiv_losses[i] = test_loss
        indiv_errors[i] = error_m
        t_test_loss += test_loss.item()
        t_f_pos += f_pos.item()
        t_f_neg += f_neg.item()
        t_error_m += error_m.item()
    losses_std = torch.std(indiv_losses).item()
    errors_std = torch.std(indiv_errors).item()
    print("")
    print("Testing Loss : "+str(t_test_loss/(TESTING_DATA_SIZE/BATCH_SIZE))+"   Error Metrics :    F_P : "+str(t_f_pos/(TESTING_DATA_SIZE/BATCH_SIZE))+"   F_N : "+str(t_f_neg/(TESTING_DATA_SIZE/BATCH_SIZE))+"   Error Metric : "+str(t_error_m/(TESTING_DATA_SIZE/BATCH_SIZE))+"   Standard Deviations :   Loss : "+str(losses_std)+"   Error : "+str(errors_std))
    
    seconds = math.floor((after_time-before_time)/1000)
    minutes = math.floor(seconds/60)
    seconds = seconds % 60
    
    print(str(NUM_EPOCHS)+" epochs took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")
    
    return model;
    

if __name__ == "__main__":
    print("Start!")
    current_milli_time = lambda: int(round(time.time() * 1000))
    before_time = current_milli_time()

    files = os.listdir(".")
    m = [int(f[9:]) for f in files if len(f) > 9 and f[0:9] == "unettrial"]
    if len(m) > 0:
        folder = "unettrial" + str(max(m) + 1)
    else:
        folder = "unettrial1"
    os.mkdir(folder)

    print("Created session folder " + folder)

    print("Loading data...")
    data = torch.load("TRIMMED_DATA.pt")
    
    after_time = current_milli_time()
    seconds = math.floor((after_time-before_time)/1000)
    minutes = math.floor(seconds/60)
    seconds = seconds % 60
    print("Data loading took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")  
    
    model = train_model(data)
    
    '''
    
    1. Tweak BCE coefficient
    2. Dropout
    3. Cutoff at certain epoch
    4. Record data
    
    
    
    
    
    
    
    '''