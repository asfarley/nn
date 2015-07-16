class NN
  
  def initialize(input_size, output_size, learning_rate)
    @weights = Array.new(output_size)
    for i in 0..(@weights.length-1)
    @weights[i] = Array.new(input_size+1).collect! { |x| rand } # +1 for bias
    end
    @L = learning_rate
  end
  
  def train(input, target)
	update_weights(weight_gradients(input, evaluate(input), target))
  end

  def evaluate(input)
    output = Array.new(@weights.length)
    for i in 0..(@weights.length-1)
      input_biased = add_bias_to_input(input)
      neuron_input = hadamard(@weights[i], input_biased)
      output[i] = sigma(neuron_input)
    end
    return output
  end
  
  def add_bias_to_input(input)
	input.clone.insert(0,1) #Clone because insert is destructive
  end
  
  def sigma(val)
    1/(1+2.71**-val)
  end
  
  def error(output, target)
	output.map.with_index{ |x,i| output[i] - target[i] }
  end
  
  def hadamard(a,b)
	h = a.zip(b).inject(0) { |result, element| result + element[0]*element[1] }
  end
  
  def weight_gradients(input, output, target)
    e = error(output, target)
    input_biased = add_bias_to_input(input)
    weight_gradients = Array.new(@weights.length)
    for i in 0..(@weights.length-1)
      weight_gradients[i] = Array.new(@weights[i].length).collect!.with_index { |x,j| input_biased[j] * e[i] }
    end
    return weight_gradients
  end
  
  def update_weights(weight_gradient)
    for i in 0..(@weights.length-1)
	  @weights[i].collect!.with_index {|x,j| x - @L*weight_gradient[i][j] }
    end
  end
  
  def train_on_set(inputs, targets, epochs)
  epochs.times { |i| train(inputs[i%inputs.length], targets[i%inputs.length]) }
  end
  
end

net = NN.new(1,1,0.1)

inputs = [[0], [1]]
outputs = [[1], [0]]

net.train_on_set(inputs, outputs, 1000)

puts "Evaluating network on input:0"
puts net.evaluate([0])

puts "Evaluating network on input:1"
puts net.evaluate([1])