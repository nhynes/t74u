require 'paths'

local buildMLP = paths.dofile(paths.thisfile('../models/mlp.lua'))

local mlp = buildMLP{nClasses=10, imgSize=28}

local input = torch.rand(3, 28, 28)
local gradOutput = torch.rand(3, 10)

mlp:forward(input)
mlp:backward(input, gradOutput)

print('Passed!')
