require 'set'

class Value
  attr_accessor :data, :children, :op, :label, :grad, :_backward

  def initialize(data:, children: [], op: '', label: '')
    @data = data
    @children = Set.new(children)
    @op = op
    @label = label
    @grad = 0.0
    @_backward = lambda {}
  end

  def +(other)
    out = Value.new(
      data: @data + other.data,
      children: [self, other],
      op: :+
    )

    out._backward = lambda do
      @grad += out.grad
      other.grad += out.grad
    end

    out
  end

  def -(other)
    self + other * Value.new(data: -1)
  end

  def *(other)
    out = Value.new(
      data: @data * other.data,
      children: [self, other],
      op: :*
    )

    out._backward = lambda do
      @grad += out.grad * other.data
      other.grad += out.grad * data
    end

    out
  end

  def /(other)
    self * other ** -1
  end

  def **(other)
    raise 'Only support scalar power' unless other.is_a?(Numeric)

    out = Value.new(
      data: @data ** other,
      children: [self],
      op: "**#{other}"
    )

    out._backward = lambda do
      @grad += out.grad * other * data ** (other - 1)
    end

    out
  end

  def tanh
    t = (Math.exp(2 * data) - 1) / (Math.exp(2 * data) + 1)
    out = Value.new data: t, children: [self], op: :tanh

    out._backward = lambda do
      @grad += out.grad * (1 - t ** 2)
    end

    out
  end

  def backward
    topo = []
    visited = Set.new

    def build_topo(v, visited, topo)
      return if visited.include?(v)

      visited.add(v)

      v.children.each { build_topo(_1, visited, topo) }
      topo << v
    end

    build_topo(self, visited, topo)
    @grad = 1.0

    topo.reverse_each { _1._backward.() }
  end

  def to_s
    "#{label}=#{data}"
  end
end
