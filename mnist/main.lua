require 'torch'
require 'optim'
require 'xlua'
require 'paths'
local argcheck = require 'argcheck'

paths.dofile(paths.thisfile('dataloader.lua'))
local models = paths.dofile(paths.thisfile('models/init.lua'))

local check = argcheck{
  pack=true,
  ------------ General options --------------------
  {name='seed', type='number', default=0, help='random seed (for reproducability)'},
  {name='dispFreq', type='number', default=1, help='show progress every n epochs'},
  ------------ Model options --------------------
  {name='modelType', type='string', default='mlp', help='mlp or conv'},
  ------------- Training options --------------------
  {name='nEpochs', type='number', default=20, help='train for this many epochs'},
  {name='batchSize', type='number', default=128, help='minibatch size'},
  ---------- Optimization options ----------------------
  {name='lr', type='number', default=0.001, help='learning rate'},
  {name='momentum', type='number', default=0.9, help='SGD momentum hyperparameter'},
  {name='weightDecay', type='number', default=1e-4, help='weight decay strength'},
}

local STATUS_STR = 'Epoch %2d | train loss: %.3f | val loss: %.3f | val acc: %.2f%%'

function main(...)
  local opt = check(...)

  opt.nClasses = 10
  opt.imgSize = 28

  local Model = models.init(opt)

  torch.manualSeed(opt.seed)

  local loader = DataLoader(opt)
  local model = Model(opt)
  local crit = nn.CrossEntropyCriterion()

  local optimState = {
    learningRate = opt.lr,
    learningRateDecay = 0.0,
    weightDecay = opt.weightDecay,
    nesterov = true,
    momentum = opt.momentum,
    dampening = 0.0,
  }

  local params, gradParams = model:getParameters()
  local function f() return crit.output, gradParams end

  local inpImages = torch.Tensor()
  local inpLabels = torch.Tensor()

  local trainSize = loader:size('train')
  local valSize = loader:size('val')

  for i=1,opt.nEpochs do
    local trainLoss = 0
    local trainBatches = 0
    for n, minibatch in loader:run() do
      inpImages:resize(minibatch.images:size()):copy(minibatch.images)
      inpLabels:resize(minibatch.labels:size()):copy(minibatch.labels)

      model:forward(inpImages)

      trainLoss = trainLoss + crit:forward(model.output, inpLabels)
      trainBatches = trainBatches + 1

      crit:backward(model.output, inpLabels)
      model:zeroGradParameters()
      model:backward(inpImages, crit.gradInput)

      optim.sgd(f, params, optimState)

      xlua.progress(n, math.ceil(trainSize / opt.batchSize))
    end

    local valLoss = 0
    local valBatches = 0
    local valCorrect = 0
    for n, minibatch in loader:run('val') do
      inpImages:resize(minibatch.images:size()):copy(minibatch.images)
      inpLabels:resize(minibatch.labels:size()):copy(minibatch.labels)

      model:forward(inpImages)

      valLoss = valLoss + crit:forward(model.output, inpLabels)
      valBatches = valBatches + 1

      local maxProb, amaxProb = model.output:max(2)
      valCorrect = valCorrect + amaxProb:eq(minibatch.labels:long()):sum()

      xlua.progress(n, math.ceil(valSize / opt.batchSize))
    end

    if i % opt.dispFreq == 0 then
      print(string.format(STATUS_STR,
        i, trainLoss/trainBatches, valLoss/valBatches, valCorrect/valSize*100))
    end
  end
end

return main
