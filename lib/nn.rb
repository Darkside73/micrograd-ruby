module NN
  class Neuron
    attr_reader :nin, :w, :b

    def initialize(nin)
      @nin = nin
      @w = Array.new(nin) { Value.new data: rand(-1.0..1.0) }
      @b = Value.new data: rand(-1.0..1.0)
    end

    def call(x)
      xv = x.map { _1.is_a?(Value) ? _1 : Value.new(data: _1) }
      act = w.zip(xv).map { _1 * _2 }.reduce(:+) + b
      act.tanh
    end

    def parameters
      w + [b]
    end
  end

  class Layer
    attr_reader :nin, :nout, :neurons

    def initialize(nin, nout)
      @nin = nin
      @nout = nout

      @neurons = Array.new(nout) { Neuron.new(nin) }
    end

    def call(x)
      out = @neurons.map { _1.(x) }
      out.count == 1 ? out.first : out
    end

    def parameters
      @neurons.map(&:parameters).flatten
    end
  end

  class MLP
    attr_reader :layers

    def initialize(nin, nouts)
      sz = [nin, *nouts]
      @layers = sz.each_cons(2).map { Layer.new(_1, _2) }
    end

    def call(x)
      @layers.each { |layer| x = layer.(x) }
      x
    end

    def parameters
      @layers.map(&:parameters).flatten
    end
  end
end
