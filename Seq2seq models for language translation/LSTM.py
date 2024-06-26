"""
LSTM model.  (c) 2021 Georgia Tech

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
import torch.nn as nn


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################

        # i_t: input gate
        self.wii= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.bii= nn.Parameter(torch.Tensor(hidden_size))
        self.whi= nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bhi= nn.Parameter(torch.Tensor(hidden_size))
        self.sigi= nn.Sigmoid()
        
        # f_t: the forget gate
        self.wif = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.bif = nn.Parameter(torch.Tensor(hidden_size))
        self.whf= nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bhf= nn.Parameter(torch.Tensor(hidden_size))
        self.sigf= nn.Sigmoid()
        

        # g_t: the cell gate
        self.wig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.big = nn.Parameter(torch.Tensor(hidden_size))
        self.whg= nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bhg= nn.Parameter(torch.Tensor(hidden_size))
        self.tanh = nn.Tanh()
        
        # o_t: the output gate
        self.wio = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.bio = nn.Parameter(torch.Tensor(hidden_size))
        self.who= nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bho= nn.Parameter(torch.Tensor(hidden_size))
        self.sigo = nn.Sigmoid()
        

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (batch, sequence, feature)"""

        batch_size, sequence, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        for t in range(sequence):
            x_t = x[:, t, :]
            i_t = self.sigi(torch.mm(x_t, self.wii) + self.bii + torch.mm(h_t, self.whi) + self.bhi)
            f_t = self.sigf(torch.mm(x_t, self.wif) + self.bif + torch.mm(h_t, self.whf) + self.bhf)
            g_t = self.tanh(torch.mm(x_t, self.wig) + self.big + torch.mm(h_t, self.whg) + self.bhg)
            o_t = self.sigo(torch.mm(x_t, self.wio) + self.bio + torch.mm(h_t, self.who) + self.bho)


            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.tanh(c_t)

           
        
            

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              #
        #   h_t and c_t should be initialized to zeros.                                #
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        #h_t, c_t = None, None  #remove this line when you start implementing your code
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
