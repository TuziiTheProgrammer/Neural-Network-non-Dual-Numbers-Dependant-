local Layer = {

	Weights = {};
	Biases = {};
	input = 0;
	output = 0;
	costWeightGrad = {};
	costBiasGrad = {}
}

Layer.__index = Layer

function Layer.new(inputNodes: number, outputNodes: number)
	local meta = setmetatable({}, Layer)

	meta.input = inputNodes
	meta.output = outputNodes
	meta.Weights = {}
	meta.Biases = {}
	meta.costBiasGrad = {}
	meta.costWeightGrad = {}
	
	for i = 1, inputNodes do
		meta.Weights[i] = {}
		meta.costWeightGrad[i] = {}
		for j = 1, outputNodes do
			meta.Weights[i][j] = (math.random() * 2 - 1) / math.sqrt(inputNodes)
			meta.costWeightGrad[i][j] = 0
		end
	end

	for j = 1, outputNodes do
		meta.Biases[j] = (math.random() * 2 - 1) / math.sqrt(inputNodes)
		meta.costBiasGrad[j] = 0
	end
	
	return meta
end


function Layer:SetGradiants(rate:number)

	for nodeOut = 1, self.output do

		self.Biases[nodeOut] -= self.costBiasGrad[nodeOut] * rate

		for nodeIn = 1, self.input do

			self.Weights[nodeIn][nodeOut] -= self.costWeightGrad[nodeIn][nodeOut] * rate

		end

	end
end


function Layer:CalculateOutputs(inputs_):{}

	local weightedOutputsToNode = {}
	
	for i = 1, self.output do
		local weigthedInput:number = self.Biases[i]
		for j = 1, self.input do
			weigthedInput += inputs_[j] * self.Weights[j][i] 
		end

		weightedOutputsToNode[i] = self.Activation(weigthedInput)

	end
	return weightedOutputsToNode
end

function Layer.Cost(activation: number, expected: number)
	local penalty = (expected == 1) and 2 or 1
	return penalty*(activation - expected) ^ 2
end




function Layer.Activation(weight: number)
	return 1/(1 + math.exp(-weight))
end

function Layer.Activation2(weight: number)
	return weight > 0 and 1 or 0
end

function Layer.Activation3(weight: number)
	return math.max(0, weight)
end

export type Layer = typeof(Layer.new())
return Layer
