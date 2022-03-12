import numpy as np
import torch
import torch.nn.functional as F

def train(model, data_loader, valid_loader, criterion, optimizer, lr_scheduler, modelpath, device, epochs):
   model.train()
   train_loss = []
   valid_loss = []
   valid_acc = []

   for epoch in range(epochs):
      print("EPOCH", epoch+1)
      avg_loss = 0.0

      for batch_num, (tweet, input_id, attention_masks, target) in enumerate(data_loader):
         if batch_num % 100 == 0:
            print("batch", batch_num)
         input_ids, attention_masks, target = input_id.to(device), attention_masks.to(device), target.to(device)
         loss, logits = model(input_ids,
                              token_type_ids=None,
                              attention_mask=attention_masks,
                              labels=target,
                              return_dict=False
                              )

         optimizer.zero_grad()
         loss.backward()
         optimizer.step()


         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
         avg_loss += loss.item()
         lr_scheduler.step()

      training_loss = avg_loss / len(data_loader)

      print('Epoch: ', epoch + 1)
      print('training loss = ', training_loss)
      train_loss.append(training_loss)

      validation_loss, top1_acc = test_classify(model, valid_loader, criterion, device)
      valid_loss.append(validation_loss)
      valid_acc.append(top1_acc)

      torch.save({
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'lr_scheduler': lr_scheduler.state_dict(),
      }, modelpath)

   return train_loss, valid_loss, valid_acc

def test_classify(model, valid_loader, criterion, device):
   model.eval()
   test_loss = []
   top1_accuracy = 0
   total = 0

   for batch_num, (tweet, input_id, attention_masks, target) in enumerate(valid_loader):
      input_ids, attention_masks, target = input_id.to(device), attention_masks.to(device), target.to(device)

      loss, logits = model(input_ids,
                           token_type_ids=None,
                           attention_mask=attention_masks,
                           labels=target,
                           return_dict=False)

      test_loss.extend([loss.item()] * input_id.size()[0])

      predictions = F.softmax(logits, dim=1)

      _, top1_pred_labels = torch.max(predictions, 1)
      top1_pred_labels = top1_pred_labels.view(-1)

      top1_accuracy += torch.sum(torch.eq(top1_pred_labels, target)).item()
      total += len(target)

   print('Validation Loss: {:.4f}\tTop 1 Validation Accuracy: {:.4f}'.format(np.mean(test_loss), top1_accuracy / total))

   return np.mean(test_loss), top1_accuracy / total