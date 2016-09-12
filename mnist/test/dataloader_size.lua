require 'paths'

paths.dofile(paths.thisfile('../dataloader.lua'))

local loader = DataLoader{batchSize=4}

local trainSize = loader:size('train')
assert(trainSize == 60000, 'Expected train size to be 60000 but was '..tostring(trainSize))

local valSize = loader:size('val')
assert(valSize == 10000, 'Expected val size to be 10000 but was '..tostring(valSize))

print 'Passed!'
