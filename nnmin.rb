class NN
  def initialize(input_size, output_size, learning_rate)
    @weights = Array.new(output_size).collect!{ Array.new(input_size+1).collect!{ |x| rand}}
    @L = learning_rate
  end
  def evaluate(input)
    output = Array.new(@weights.length)
    for i in 0..(@weights.length-1)
	  neuron_input = @weights[i].zip(input.clone.insert(0,1)).inject(0) { |result, element| result + element[0]*element[1] }
      output[i] = 1/(1+2.71**-neuron_input)
    end
    return output
  end
  def weight_gradients(input, output, target)
	e = output.map.with_index{ |x,i| output[i] - target[i] }
    weight_gradients = Array.new(@weights.length)
    for i in 0..(@weights.length-1)
      weight_gradients[i] = Array.new(@weights[i].length).collect!.with_index { |x,j| input.clone.insert(0,1)[j] * e[i] }
    end
    return weight_gradients
  end
  def update_weights(weight_gradient)
	 @weights.each_with_index{ |x,i| x.collect!.with_index {|x,j| x - @L*weight_gradient[i][j] } } 
  end
  def train_on_set(inputs, targets, epochs)
  epochs.times{|i| update_weights(weight_gradients(inputs[i%inputs.length], evaluate(inputs[i%inputs.length]), targets[i%inputs.length])) }
  end
end

net = NN.new(1,1,0.1)

inputs = [[0], [1]]
outputs = [[1], [0]]

net.train_on_set(inputs, outputs, 1000)
puts "Evaluating network on input:0 -> #{net.evaluate([0])}"
puts "Evaluating network on input:1 -> #{net.evaluate([1])}"