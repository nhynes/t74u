require 'nn'
require 'paths'

local buildCNN = paths.dofile(paths.thisfile('../models/conv.lua'))

local cnn = buildCNN{nClasses=10, imgSize=28}

local input = torch.rand(3, 1, 28, 28)
local gradOutput = torch.rand(3, 10)

cnn:forward(input)
cnn:backward(input, gradOutput)

print('Passed!')
