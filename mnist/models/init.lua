require 'nn'

local models = {}

function models.init(opt)
  return paths.dofile(paths.thisfile(opt.modelType..'.lua'))
end

return models
