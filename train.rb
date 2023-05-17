require_relative 'lib/value'
require_relative 'lib/nn'
require 'optparse'

options = {}
OptionParser.new do |opt|
  opt.on('--loops 20') { |o| options[:loops] = o }
  opt.on('--lrate 0.3') { |o| options[:lrate] = o }
end.parse!

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0].map { Value.new data: _1 }

n = NN::MLP.new(3, [4, 4, 1])

puts '--desired--'
puts ys.map(&:data)

loops = options[:loops].to_i
loops = 20 if loops.zero?
lrate = options[:lrate].to_f
lrate = 0.3 if lrate.zero?

puts '--training loop--'

ypred = nil
loops.times do |i|
  step = i + 1

  # forward pass
  ypred = xs.map { |x| n.(x) }
  loss = ys.zip(ypred).map do |ygt, yout|
    (ygt - yout) ** 2
  end.reduce(:+)

  # backward pass
  n.parameters.each { _1.grad = 0.0 }
  loss.backward

  # update parameters (gradient descent)
  n.parameters.each { _1.data += -1 * lrate * _1.grad }

  puts "step=#{step}, loss=#{loss.data}"
end

puts '--prediction--'
puts ypred.map(&:data)
