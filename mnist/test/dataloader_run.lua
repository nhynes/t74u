require 'paths'

paths.dofile(paths.thisfile('../dataloader.lua'))

torch.manualSeed(42)

local loader = DataLoader{batchSize=4}

local expectedImgSums = {135940, 87747, 124043}
local expectedLabels = {
  torch.ByteTensor{7, 8, 7, 10},
  torch.ByteTensor{5, 3, 2, 4},
  torch.ByteTensor{9, 10, 8, 1},
}

local trainLoader = loader:run('train')
for i=1,#expectedImgSums do
  local n, mb = trainLoader()
  assert(n == i, 'Wrong minibatch number returned...')
  assert(mb.images:size(1) == 4,
    'Expected 4 images but got '..mb.images:size(1)..'. Are you slicing correctly?')
  assert(mb.labels:size(1) == 4,
    'Expected 4 labels but got '..mb.labels:size(1)..'. Are you slicing correctly?')
  assert(mb.images:sum() == expectedImgSums[i],
    'Incorrect images returned for minibatch '..'. Are you slicing correctly?')
  assert(mb.labels:eq(expectedLabels[i]),
    'Incorrect labels returned for minibatch '..'. Are you slicing correctly?')
end

loader = DataLoader{batchSize=10000}
local valLoader = loader:run('val')

local n, b = valLoader()
assert(b.images:size(1) == 10000)

assert(valLoader() == nil,
  'Loader returned too much data! Did you select the correct partition?')

print('Passed!')
