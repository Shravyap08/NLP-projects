"""
S2S Decoder model.  (c) 2021 Georgia Tech

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

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        
                
        
        self.embedding = nn.Embedding(output_size, emb_size)
        
        if model_type == 'RNN':
            self.rnn= nn.RNN(emb_size, encoder_hidden_size, batch_first=True)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(emb_size, encoder_hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        
            
        self.fc= nn.Linear(decoder_hidden_size, output_size)
        self.logsoftmax= nn.LogSoftmax(dim=1)
        
        
        self.dropout = nn.Dropout(dropout)

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
                
        """
        q = hidden.squeeze(0)
        
        K= encoder_outputs
       
        q_norm = torch.norm(q, dim=1).unsqueeze(1)
        
        K_norm = torch.norm(K, dim=2)
        
        q_K = torch.bmm(K, q.unsqueeze(2)).squeeze(2) 
       
        
        cosine_similarity = q_K / (q_norm * K_norm)
        
       
       
        attention = torch.softmax(cosine_similarity, dim=1).unsqueeze(1)
        
        return attention
        
       
        
        
        
       

        
       

        

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        # It is recommended that you implement the cosine similarity function from  #
        # the formula given in the docstring. This exercise will build up your     #
        # skills in implementing mathematical formulas working with tensors.        #
        # Alternatively you may use nn.torch.functional.cosine_similarity or        #
        # some other similar function for your implementation.                      #
        #############################################################################

        #attention = None   #remove this line when you start implementing your code
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
    def forward(self, input, hidden, encoder_outputs=None, attention=False):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden state of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the state coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
                where N is the batch size, T is the sequence length
        """
        
        
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
       
        if attention:
            
            if self.model_type == 'LSTM':
                hn, cn = hidden
                attn_weights_h = self.compute_attention(hn, encoder_outputs)
                attn_weights_c = self.compute_attention(cn, encoder_outputs)
                
                context_vector_h = torch.bmm(attn_weights_h, encoder_outputs)
                context_vector_c = torch.bmm(attn_weights_c, encoder_outputs)
                
                context_vector = (context_vector_h.permute(1,0,2), context_vector_c.permute(1,0,2))
                hidden_new = context_vector
                
                #hidden_new = context_vector.permute(1,0,2)
               
                output, hidden = self.rnn(embedded, hidden_new)
                
                
                
            else:
                attn_weights = self.compute_attention(hidden, encoder_outputs)
                
                context_vector = torch.bmm(attn_weights, encoder_outputs)
                
                hidden_new = context_vector.permute(1,0,2)
                
                output, hidden = self.rnn(embedded, hidden_new)
                
                
            
                
        else:
            
            output, hidden = self.rnn(embedded, hidden)
            
            
      
                             
        output = self.fc(output.squeeze(1))
        
        output = self.logsoftmax(output)
        
        
           
            
           
        
        
             
        
        
        
        
        #####################################
        
       
        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #                                                                           #
        #       If attention is true, compute the attention probabilities and use   #
        #       them to do a weighted average on the encoder_outputs to determine   #
        #       the hidden (and cell if LSTM) states that will be consumed by the   #
        #       recurrent layer.                                                    #
        #                                                                           #
        #       Apply linear layer and log-softmax activation to output tensor      #
        #       before returning it.                                                #
        #############################################################################

        #output, hidden = None, None     #remove this line when you start implementing your code

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
