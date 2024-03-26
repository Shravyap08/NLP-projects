"""
Transformer model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        
        
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.embeddingL = None      #initialize word embedding layer
        self.posembeddingL = None   #initialize positional embedding layer
        
        
        
        self.embeddingL = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_dim)
        self.posembeddingL = nn.Embedding(num_embeddings=max_length, embedding_dim=hidden_dim)
        
        
        
        

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        self.feed_forward_1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.feed_forward_2 = nn.Linear(self.dim_feedforward, self.hidden_dim)
        self.relu_act = nn.ReLU()
        self.norm_ff = nn.LayerNorm(self.hidden_dim)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
    
        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.fl = nn.Linear(self.hidden_dim, self.output_size)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """
        
        embeddings = self.embed(inputs)
        attn_outputs = self.multi_head_attention(embeddings)
        ff_outputs = self.feedforward_layer(attn_outputs)
        outputs = self.final_layer(ff_outputs)

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling TransformerTranslator class methods here.      #
        #############################################################################
        #outputs = None      #remove this line when you start implementing your code
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################
      
        #embeddings = None       #remove this line when you start implementing your code
        
        word_embeddings = self.embeddingL(inputs) 
        positions = torch.arange(inputs.size(1), device=self.device).unsqueeze(0).repeat(inputs.size(0), 1)
        position_embeddings = self.posembeddingL(positions)
        embeddings = word_embeddings + position_embeddings
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        
        k1 = self.k1(inputs)
        q1 = self.q1(inputs)
        v1 = self.v1(inputs)
        
        k2 = self.k2(inputs)
        q2 = self.q2(inputs)
        v2 = self.v2(inputs)
        
        sf = torch.sqrt(torch.tensor(self.dim_k, dtype=torch.float32))
        
        attn_scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / sf
        attn_scores1 = self.softmax(attn_scores1)
        output_head1 = torch.matmul(attn_scores1, v1)

        attn_scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / sf
        attn_scores2 = self.softmax(attn_scores2)
        output_head2 = torch.matmul(attn_scores2, v2)

        
        multi_head_output = torch.cat((output_head1, output_head2), dim=-1)
        
        outputs = self.attention_head_projection(multi_head_output)
        outputs = self.norm_mh(outputs + inputs)
        
        #outputs = None      #remove this line when you start implementing your code
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        
        out = self.feed_forward_1(inputs)
        out = self.relu_act(out)
        out = self.feed_forward_2(out)
        out = out + inputs
        outputs = self.norm_ff(out)
    
        #outputs = None      #remove this line when you start implementing your code
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################
        outputs =self.fl(inputs)
        
        #outputs = None      #remove this line when you start implementing your code
                
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

class FullTransformerTranslator(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, num_layers_enc=2, num_layers_dec=2, dropout=0.2, max_length=43, ignore_index=1):
        super(FullTransformerTranslator, self).__init__()

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx=ignore_index

        seed_torch(0)

        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the Transformer Layer          #
        # You should use nn.Transformer                                              #
        ##############################################################################
        
        self.transformer= nn.Transformer(d_model= self.hidden_dim, nhead= self.num_heads, 
                                         num_encoder_layers= num_layers_enc,
                                         num_decoder_layers= num_layers_dec,
                                         dim_feedforward = self.dim_feedforward,
                                         dropout = dropout, 
                                         activation='relu',
                                         batch_first=True,
                                         custom_encoder=None,
                                         custom_decoder=None
                                         )
        
        
        ##############################################################################
        # TODO:
        # Deliverable 2: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Initialize embeddings in order shown below.                                #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.srcembeddingL = None       #embedding for src
        self.tgtembeddingL = None       #embedding for target
        self.srcposembeddingL = None    #embedding for src positional encoding
        self.tgtposembeddingL = None    #embedding for target positional encoding
        
        self.srcembeddingL = nn.Embedding(num_embeddings=self.input_size, embedding_dim=hidden_dim,padding_idx = None)
        self.tgtembeddingL = nn.Embedding(num_embeddings=self.input_size, embedding_dim=hidden_dim,padding_idx = None)
        
       
        
        
        self.srcposembeddingL = nn.Embedding(num_embeddings= max_length, embedding_dim=hidden_dim)
        self.tgtposembeddingL = nn.Embedding(num_embeddings= max_length, embedding_dim=hidden_dim)
        
        #self.srcembeddingL = self.srcembeddingL.to(device)
        #self.tgtembeddingL = self.tgtembeddingL.to(device)
        #self.srcposembeddingL = self.srcposembeddingL.to(device)
        #self.tgtposembeddingL = self.tgtposembeddingL.to(device)
        
        
        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the final layer.               #
        ##############################################################################
        
        self.fl = nn.Linear(self.hidden_dim, self.output_size)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, src, tgt):
        """
         This function computes the full Transformer forward pass used during training.
         Put together all of the layers you've developed in the correct order.

         :param src: a PyTorch tensor of shape (N,T) these are tokenized input sentences
                tgt: a PyTorch tensor of shape (N,T) these are tokenized translations
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the full Transformer stack for the forward pass. #
        #############################################################################
        outputs=None
        # shift tgt to right, add one <sos> to the beginning and shift the other tokens to right
      
        tgt = self.add_start_token(tgt)
       
        # embed src and tgt for processing by transformer
        
        src_embeddings = self.srcembeddingL(src) 
        tgt_embeddings = self.tgtembeddingL(tgt) 

        src_embeddings += self.srcposembeddingL(torch.arange(0, src.size(1), device=self.device).unsqueeze(0).expand(src.size(0), -1))
        tgt_embeddings += self.tgtposembeddingL(torch.arange(0, tgt.size(1), device=self.device).unsqueeze(0).expand(tgt.size(0), -1))

        
        # create target mask and target key padding mask for decoder
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
        tgt_mask = tgt_mask == float('-inf')
        
        tgt_key_padding_mask = (tgt == self.pad_idx)
        
        # invoke transformer to generate output
        transformer_output = self.transformer(
            src=src_embeddings,
            tgt=tgt_embeddings,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
       
        # pass through final layer to generate outputs
        outputs = self.fl(transformer_output)
        

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def generate_translation(self, src):
        """
         This function generates the output of the transformer taking src as its input
         it is assumed that the model is trained. The output would be the translation
         of the input

         :param src: a PyTorch tensor of shape (N,T)

         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 5: You will be calling the transformer forward function to    #
        # generate the translation for the input.                                   #
        #############################################################################
        #outputs = None
          
        
        batch_size = src.size(0)
        max_length = self.max_length
        
       
        tgt = torch.full((batch_size, max_length),self.pad_idx, dtype=torch.long, device=self.device)
        #outputs = torch.full((batch_size, max_length, self.output_size), self.pad_idx, dtype=torch.long, device=self.device)
        outputs = torch.zeros((batch_size, max_length, self.output_size), device=self.device)

        tgt[:, 0] = src[:, 0]
        
        for t in range(1,max_length):
            output = self.forward(src, tgt)
            next_token = output.argmax(dim=2)[:, t-1]
            
            if t < max_length: 
                tgt[:, t] = next_token
            outputs[:, t-1, :] = output[:, t-1, :]
                
                
        output = self.forward(src, tgt)
        outputs[:, -1, :] = output[:, -1, :]
       
           
            
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def add_start_token(self, batch_sequences, start_token=2):
        """
            add start_token to the beginning of batch_sequence and shift other tokens to the right
            if batch_sequences starts with two consequtive <sos> tokens, return the original batch_sequence

            example1:
            batch_sequence = [[<sos>, 5,6,7]]
            returns:
                [[<sos>,<sos>, 5,6]]

            example2:
            batch_sequence = [[<sos>, <sos>, 5,6,7]]
            returns:
                [[<sos>, <sos>, 5,6,7]]
        """
        def has_consecutive_start_tokens(tensor, start_token):
            """
                return True if the tensor has two consecutive start tokens
            """
            consecutive_start_tokens = torch.tensor([start_token, start_token], dtype=tensor.dtype,
                                                    device=tensor.device)

            # Check if the first two tokens in each sequence are equal to consecutive start tokens
            is_consecutive_start_tokens = torch.all(tensor[:, :2] == consecutive_start_tokens, dim=1)

            # Return True if all sequences have two consecutive start tokens at the beginning
            return torch.all(is_consecutive_start_tokens).item()

        if has_consecutive_start_tokens(batch_sequences, start_token):
            return batch_sequences

        # Clone the input tensor to avoid modifying the original data
        modified_sequences = batch_sequences.clone()

        # Create a tensor with the start token and reshape it to match the shape of the input tensor
        start_token_tensor = torch.tensor(start_token, dtype=modified_sequences.dtype, device=modified_sequences.device)
        start_token_tensor = start_token_tensor.view(1, -1)

        # Shift the words to the right
        modified_sequences[:, 1:] = batch_sequences[:, :-1]

        # Add the start token to the first word in each sequence
        modified_sequences[:, 0] = start_token_tensor

        return modified_sequences

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True