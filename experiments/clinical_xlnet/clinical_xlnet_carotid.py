# https://towardsdatascience.com/multi-label-text-classification-with-xlnet-b5f5755302df
# https://colab.research.google.com/drive/1o3cv-YSPGiKftCvFnCiMcygARqdaxrM7
import os
import math
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# pip install transformers==2.4.1
from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import pickle
import time
import argparse

class XLNetForMultiLabelSequenceClassification(torch.nn.Module):

    def __init__(self, num_labels=2):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        # initialize xlnet config
        config = XLNetConfig.from_pretrained('D:\Projects\carodit_nlp\experiments\clinical_xlnet\pretrained_model\config.json', num_labels=num_labels)
        self.xlnet = XLNetModel(config)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_labels)

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None):
        # last hidden layer
        last_hidden_state = self.xlnet(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        # pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_xlnet_decoder(self):
        """
        Freeze XLNet weight parameters. They will not be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = False

    def unfreeze_xlnet_decoder(self):
        """
        Unfreeze XLNet weight parameters. They will be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state


def train(model, num_epochs, optimizer, train_dataloader, valid_dataloader,
          model_save_path, train_loss_set=[], valid_loss_set=[],
          lowest_eval_loss=None, start_epoch=0, device="cpu"):
    """
      Train the model and save the model with the lowest validation loss
    """
    model.to(device)

    # trange is a tqdm wrapper around the normal python range
    for i in trange(num_epochs, desc="Epoch"):
        # if continue training from saved model
        actual_epoch = start_epoch + i

        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        # Tracking variables
        tr_loss = 0
        num_train_samples = 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            # store train loss
            tr_loss += loss.item()
            num_train_samples += b_labels.size(0)
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            #scheduler.step()

        # Update tracking variables
        epoch_train_loss = tr_loss/num_train_samples
        train_loss_set.append(epoch_train_loss)

        print("Train loss: {}".format(epoch_train_loss))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss = 0
        num_eval_samples = 0

        # Evaluate data for one epoch
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate validation loss
                loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                # store valid loss
                eval_loss += loss.item()
                num_eval_samples += b_labels.size(0)

        epoch_eval_loss = eval_loss/num_eval_samples
        valid_loss_set.append(epoch_eval_loss)
        print("Valid loss: {}".format(epoch_eval_loss))

        if lowest_eval_loss == None:
            lowest_eval_loss = epoch_eval_loss
            # save model
            save_model(model, model_save_path, actual_epoch,
                       lowest_eval_loss, train_loss_set, valid_loss_set)
        else:
            if epoch_eval_loss < lowest_eval_loss:
                lowest_eval_loss = epoch_eval_loss
                # save model
                save_model(model, model_save_path, actual_epoch,
                           lowest_eval_loss, train_loss_set, valid_loss_set)
        print("\n")
    return model, train_loss_set, valid_loss_set


def generate_predictions(model, df, num_labels, device="cpu", batch_size=32):
    num_iter = math.ceil(df.shape[0] / batch_size)

    pred_probs = np.array([]).reshape(0, num_labels)

    model.to(device)
    model.eval()

    for i in range(num_iter):
        df_subset = df.iloc[i * batch_size:(i + 1) * batch_size, :]
        X = df_subset["features"].values.tolist()
        masks = df_subset["masks"].values.tolist()
        X = torch.tensor(X)
        masks = torch.tensor(masks, dtype=torch.long)
        X = X.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            logits = model(input_ids=X, attention_mask=masks)
            logits = logits.sigmoid().detach().cpu().numpy()
            pred_probs = np.vstack([pred_probs, logits])

    return pred_probs


def save_model(model, save_path, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist):
    """
    Save the model to the path directory provided
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {'epochs': epochs,
                  'lowest_eval_loss': lowest_eval_loss,
                  'state_dict': model_to_save.state_dict(),
                  'train_loss_hist': train_loss_hist,
                  'valid_loss_hist': valid_loss_hist
                  }
    torch.save(checkpoint, save_path)
    print("Saving model at epoch {} with validation loss of {}".format(epochs, lowest_eval_loss))
    return


def load_model(save_path):
    """
    Load the model from the path directory provided
    """
    checkpoint = torch.load(save_path)
    model_state_dict = checkpoint['state_dict']
    model = XLNetForMultiLabelSequenceClassification(num_labels=model_state_dict["classifier.weight"].size()[0])
    model.load_state_dict(model_state_dict)
    epochs = checkpoint["epochs"]
    lowest_eval_loss = checkpoint["lowest_eval_loss"]
    train_loss_hist = checkpoint["train_loss_hist"]
    valid_loss_hist = checkpoint["valid_loss_hist"]
    return model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist


def create_attn_masks(input_ids):
    """
    Create attention masks to tell model whether attention should be applied to
    the input id tokens. Do not want to perform attention on padding tokens.
    """
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks


def tokenize_inputs(text_list, tokenizer, num_embeddings=512):
    """
    Tokenizes the input text input into ids. Appends the appropriate special
    characters to the end of the text to denote end of sentence. Truncate or pad
    the appropriate sequence length.
    """
    # tokenize the text, then truncate sequence to the desired length minus 2 for
    # the 2 special characters
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings-2], text_list))
    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # append special token "<s>" and </s> to end of sentence
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # pad sequences
    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
    return input_ids


def create_training_testing_data_internal(file_path, n, num_embeddings):
    train_data = pd.read_csv(os.path.join(file_path, 'round_'+n, 'training_'+n+'.csv'))
    test_data = pd.read_csv(os.path.join(file_path, 'round_'+n, 'testing_'+n+'.csv'))
    label_cols = list(train_data.columns)[2:]

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=False)
    train_text_list = train_data["processed_content"].values
    test_text_list = test_data["processed_content"].values
    train_input_ids = tokenize_inputs(train_text_list, tokenizer, num_embeddings=num_embeddings)
    test_input_ids = tokenize_inputs(test_text_list, tokenizer, num_embeddings=num_embeddings)

    train_attention_masks = create_attn_masks(train_input_ids)
    test_attention_masks = create_attn_masks(test_input_ids)
    # add input ids and attention masks to the dataframe
    train_data["features"] = train_input_ids.tolist()
    train_data["masks"] = train_attention_masks

    test_data["features"] = test_input_ids.tolist()
    test_data["masks"] = test_attention_masks

    return train_data, test_data, label_cols


def create_training_testing_data_external(file_path, num_embeddings):
    test_data = pd.read_csv(os.path.join(file_path, 'testing.csv'))
    label_cols = list(test_data.columns)[2:]
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=False)
    test_text_list = test_data["processed_content"].values
    test_input_ids = tokenize_inputs(test_text_list, tokenizer, num_embeddings=num_embeddings)
    test_attention_masks = create_attn_masks(test_input_ids)
    # add input ids and attention masks to the dataframe
    test_data["features"] = test_input_ids.tolist()
    test_data["masks"] = test_attention_masks
    return test_data, label_cols


def get_train_validation_dataloader(batch_size, train, label_cols):
    train, valid = train_test_split(train, test_size=0.1, random_state=42)

    X_train = train["features"].values.tolist()
    X_valid = valid["features"].values.tolist()

    train_masks = train["masks"].values.tolist()
    valid_masks = valid["masks"].values.tolist()


    Y_train = train[label_cols].values.tolist()
    Y_valid = valid[label_cols].values.tolist()

    X_train = torch.tensor(X_train)
    X_valid = torch.tensor(X_valid)

    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_valid = torch.tensor(Y_valid, dtype=torch.float32)

    train_masks = torch.tensor(train_masks, dtype=torch.long)
    valid_masks = torch.tensor(valid_masks, dtype=torch.long)
    # Create an iterator of our data with torch DataLoader. This helps save on
    # memory during training because, unlike a for loop,
    # with an iterator the entire dataset does not need to be loaded into memory
    train_data = TensorDataset(X_train, train_masks, Y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=batch_size)

    validation_data = TensorDataset(X_valid, valid_masks, Y_valid)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data,
                                       sampler=validation_sampler,
                                       batch_size=batch_size)
    return train_dataloader, validation_dataloader


def model_testing(trained_model, test_data, label_cols, exin, round_n):
    pred_probs = generate_predictions(trained_model, test_data, len(label_cols), device="cuda", batch_size=batch_size)
    for index, elem in enumerate(label_cols):
        test_data[elem+'_pred'] = pred_probs[:, index]
    if exin == 'internal':
        with open(os.path.join('results', exin, 'round_' + round_n, 'predict_result.pickle'), 'wb') as file_pi:
            pickle.dump(test_data, file_pi)
    else:
        with open(os.path.join('results', exin, 'predict_result_'+round_n+'.pickle'), 'wb') as file_pi:
            pickle.dump(test_data, file_pi)


def model_training(train_data, label_cols, round_n):
    train_dataloader, validation_dataloader = get_train_validation_dataloader(batch_size, train_data, label_cols)

    model_save_path = output_model_file = os.path.join('models', 'round_'+round_n)

    model = XLNetForMultiLabelSequenceClassification(num_labels=len(label_cols))
    optimizer = AdamW(model.parameters(), lr=5e-6, weight_decay=0.01, correct_bias=False) # 1e-5, 2e-5 so bad...
    start = time.time()
    model, train_loss_set, valid_loss_set = train(model=model,
                                                  num_epochs=num_epochs,
                                                  optimizer=optimizer,
                                                  train_dataloader=train_dataloader,
                                                  valid_dataloader=validation_dataloader,
                                                  model_save_path=model_save_path,
                                                  device="cuda")

    # if train Model From Previous Checkpoint
    # model, start_epoch, lowest_eval_loss, train_loss_hist, valid_loss_hist = load_model(model_save_path)
    # optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, correct_bias=False)
    # model, train_loss_set, valid_loss_set = train(model=model,
    #                                               num_epochs=num_epochs,
    #                                               optimizer=optimizer,
    #                                               train_dataloader=train_dataloader,
    #                                               valid_dataloader=validation_dataloader,
    #                                               model_save_path=model_save_path,
    #                                               train_loss_set=train_loss_hist,
    #                                               valid_loss_set=valid_loss_hist,
    #                                               lowest_eval_loss=lowest_eval_loss,
    #                                               start_epoch=start_epoch,
    #                                               device="cuda")
    end = time.time()
    elapse = end - start
    with open(os.path.join('results', 'internal', 'round_'+round_n, 'elapse_time.pickle'), 'wb') as file_pi:
        pickle.dump(elapse, file_pi)
    return model, train_loss_set, valid_loss_set


def setup_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        required=True,
                        help="batch_size")
    parser.add_argument("--num_epochs",
                        default=None,
                        type=int,
                        required=True,
                        help="number of epochs")
    parser.add_argument("--num_embedding",
                        default=None,
                        type=int,
                        required=True,
                        help="number of embedding")
    parser.add_argument("--ex_in",
                        default=None,
                        type=str,
                        required=True,
                        help="internal or external ?")

    parser.add_argument("--round_num",
                        default=None,
                        type=str,
                        required=True,
                        help="which round")
    return parser


if __name__ == '__main__':
    # batch_size = 32
    # num_epochs = 3
    # ex_in = 'external'
    # round_num = '0'
    # rembedding_num = 512
    parser = setup_parser()
    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    ex_in = args.ex_in
    round_num = args.round_num
    rembedding_num = args.num_embedding

    if ex_in == 'internal':
        train_dataset, test_dataset, label_names = create_training_testing_data_internal(os.path.join('..', 'data', 'internal'), round_num, rembedding_num)
        # train_dataset = train_dataset.head(10)
        model, train_loss_set, valid_loss_set = model_training(train_dataset, label_names, round_num)
        model_testing(model, test_dataset, label_names, ex_in, round_num)
    elif ex_in == 'external':
        test_dataset, label_names = create_training_testing_data_external(os.path.join('..', 'data', 'external'), rembedding_num)
        model, start_epoch, lowest_eval_loss, train_loss_hist, valid_loss_hist = load_model(os.path.join('models', 'round_'+round_num))
        model_testing(model, test_dataset, label_names, ex_in, round_num)
    else:
        print('internal or external ?')
