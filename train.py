import numpy as np
import torch.optim as optim
import torch
import time, copy
import tqdm
from tqdm import tnrange, tqdm_notebook
import torch.nn.functional as F

# We define the model that we will use to train our algorithm.
def train_model(model, params, optim, scheduler, progress_bars=True, perclass_stat=False, weights=None, num_epochs=25):
    
    weights = torch.ones([model.n_class]) if weights is None else weights
    if not isinstance(weights, torch.Tensor):
        raise Exception("Weight vector should be a torch.Tensor")
    if weights.size(0) != model.n_class:
        raise Exception("Weight vector length does not match n_class")
    if len(params['loaders']) > 2:
        raise Exception("Too many elements in params['loaders']")
    timer = time.time()
    
    
    batch_size = params['batch_size']
    datasets = params['datasets']
    loaders = params['loaders']
    device = params['device']
    stats_path = params['stats_path']
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    stats = ""
    
    for epoch in range(num_epochs):
        
        print('-' * 80)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
        train_loss = 0
        train_acc = 0

        # Each epoch has a training and validation phase
        for phase in ['train', 'eval']:
            if phase == 'train':
                if progress_bars:
                    epoch_log = tqdm.tqdm_notebook(total=len(loaders[0])/batch_size, desc='Epoch ' + str(epoch+1), position=2)
                dataset_loader = loaders[0]
                dataset = datasets[0]
                model.train()  # Set model to training mode
                
            elif phase == 'eval':
                if progress_bars:
                    epoch_log = tqdm.tqdm_notebook(total=len(loaders[1])/batch_size, desc='Eval ', position=0)
                dataset_loader = loaders[1]
                dataset = datasets[1]
                model.eval()
                
            # Set the initial running cost and running accuracy to zero.  We will update these after each mini batch and will
            # obtain an overall running cost and accuracy for each epoch.  We want the loss to be as low as possible and the
            # accuracy to be as high as possible.
            running_loss = 0.0
            running_corrects = 0
            
            # Create dict for per class stats only if users wants it
            perclass_count = {}
            if perclass_stat:
                for class_i in range(model.n_class):
                    perclass_count[class_i] = {"total":0, "TP":0, "FP":0, "FN": 0}
        
            ## Batch processing
            for inputs, targets in dataset_loader:
                targets = targets.type(torch.LongTensor).to(device)
                inputs = inputs.to(device)

                # zero the parameter gradients
                optim.zero_grad()

                # Forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs)
                    _, prediction = torch.max(output, 1)
                    
                    # Define the loss
                    loss = F.cross_entropy(output, targets, weights)

                    # Backward pass + optimisation only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optim.step()


                ## Stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(prediction == targets)
                
                if perclass_stat:
                    for class_i in range(model.n_class):
                        ## Get indices
                        prediction_indices_pos = torch.where(prediction == class_i)
                        prediction_indices_neg = torch.where(prediction != class_i)
                        target_indices = torch.where(targets == class_i)
                        
                        ## Get variables
                        total = torch.sum(targets == class_i).item()
                        TP = torch.sum(prediction[target_indices] == targets[target_indices]).item()
                        FP = torch.sum(prediction[prediction_indices_pos] != targets[prediction_indices_pos]).item()
                        FN = torch.sum(prediction[prediction_indices_neg] != targets[prediction_indices_neg]).item()

                        # Update counts
                        perclass_count[class_i]["total"] += total
                        perclass_count[class_i]["TP"] += TP
                        perclass_count[class_i]["FP"] += FP
                        perclass_count[class_i]["FN"] += FN

                
                if progress_bars:
                    epoch_log.update(1)

            
            epoch_loss = running_loss / len(dataset)
            epoch_acc = running_corrects.double() / (len(dataset) * targets.shape[1] * targets.shape[2])
            
            perclass_stat_dict = {}
            if perclass_stat:
                for class_i in range(model.n_class):
                    perclass_stat_dict[class_i] = {}
                    
                    total = perclass_count[class_i]["total"]
                    TP_all = perclass_count[class_i]["TP"]
                    FP_all = perclass_count[class_i]["FP"]
                    FN_all = perclass_count[class_i]["FN"]
                    
                    precision = TP_all / (TP_all + FP_all) if (TP_all + FP_all) != 0 else 0
                    recall = TP_all / (TP_all + FN_all) if (TP_all + FN_all) != 0 else 0
                    
                    perclass_stat_dict[class_i]["Total"] = total
                    perclass_stat_dict[class_i]["Prec"] = precision
                    perclass_stat_dict[class_i]["Recall"] = recall
                    
                    
                    if (precision + recall) != 0:
                        perclass_stat_dict[class_i]["F-measure"] = 2 * ((precision * recall)/ (precision + recall))
                    else:
                        perclass_stat_dict[class_i]["F-measure"] = 0
                    
                    
            
            
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            extra_measures = ""
            if perclass_stat:
                for class_i in range(model.n_class):
                    temp_str = ("\t\t Class: {0} "
                                "Total: {1} "
                                "Precision: {2} "
                                "Recall: {3} "
                                "F-measure: {4} \n").format(class_i,
                                                            perclass_stat_dict[class_i]["Total"],
                                                            round(perclass_stat_dict[class_i]["Prec"], 4),
                                                            round(perclass_stat_dict[class_i]["Recall"], 4),
                                                            round(perclass_stat_dict[class_i]["F-measure"], 4))
                    extra_measures += ''.join(temp_str)
                print(extra_measures + "\n")
            
            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc
                scheduler.step()
            
            else:
                if epoch_loss < best_loss:
                    stats = ("weights: {0},"
                     "epoch {1}, "
                     "train loss: {2}, "
                     "train acc: {3}, "
                     "eval loss: {4}, "
                     "eval acc: {5}").format(weights.tolist(),
                                             epoch,
                                             np.round(train_loss, 4),
                                             np.round(train_acc.cpu().numpy(), 4),
                                             np.round(epoch_loss, 4),
                                             np.round(epoch_acc.cpu().numpy(), 4))
                    
                    print("Saving new best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    f = open(stats_path, "a+")
                    f.write(stats + "\n")
                    
                    if perclass_stat:
                        f.write(extra_measures + "\n")
                    f.close()


    time_elapsed = time.time() - timer
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            

    # Load best model and return it
    model.load_state_dict(best_model_wts)
    return model