function mlp(opt)
  -- Returns a Module that takes an Nx1x28x28 Tensor and produces a Nx10 Tensor

  local nPixels = opt.imgSize * opt.imgSize
  local nClasses = opt.nClasses

  local model = nil -- TODO: construct a model that satisfies the specification

  return model
end

return mlp
