local Layer = require(game:GetService("ReplicatedStorage").Layer)
local count = 0
type Cartesian = {x:number, y:number}
local test:Cartesian

local NeuralNetwork = {
	layer = {};
	Name = ""
}

NeuralNetwork.__index = NeuralNetwork

function NeuralNetwork:New(...:|any)
	count+=1
	local self = setmetatable({}, NeuralNetwork)
	self.Name = "Brain: "..count
	self.layer = {}
	local paramToArray = {...}

	for i = 1, #paramToArray - 1 do
		self.layer[i] = Layer.new(paramToArray[i], paramToArray[i + 1])
	end
	print("Neural Network Started")
	
	return self
end

function NeuralNetwork:CalculateNeuralOutput(inputs: {})
	
	for _,layer in self.layer do
		
		inputs = layer:CalculateOutputs(inputs)
		
	end
	
	return inputs
	
end

function NeuralNetwork:Classify(inputs:{})
	
	local outputs = self:CalculateNeuralOutput(inputs)
	--print(outputs)
	local highest = outputs[1]  
	local classification = 1


	
	for i = 2, #outputs do
		if outputs[i] > highest then
			highest = outputs[i]  
			classification = i  
		end
	end
	local adjustedClassifyer = classification - 1
	
	
	
	return {Class=adjustedClassifyer, Outputs=outputs}
		
end

function NeuralNetwork:Loss(inputPoint:{}, expectedOutput:{})
	
	
	--print(inputPoint)
	
	--print(expectedOutput)
	
	local outputs = self:CalculateNeuralOutput(inputPoint)
	--print(outputs)
	local layer = self.layer[#self.layer]
	
	local cost = 0
	
	for i = 1, #outputs do
		--print(outputs)
		cost += layer.Cost(outputs[i], expectedOutput[i])
	end	
	
	return cost
end

function NeuralNetwork:Cost(objectList:{})
	
	local cost = 0
	
	
	
	for _, thing in objectList do
		
		cost += self:Loss(thing.DangerPoints.Graph, thing.DangerPoints.expectedOutput)
	end
	local after = cost/#objectList
	
	return after
end

function NeuralNetwork:Learn(dataset:{}, rate: number)
	
	local h = 0.0001
	local orginalCost = self:Cost(dataset)
	
	for _, layer in self.layer do
		for nodeIn = 1, layer.input do
			for nodeOut = 1, layer.output do
				layer.Weights[nodeIn][nodeOut] += h
				local differentialChange = self:Cost(dataset) - orginalCost
				layer.Weights[nodeIn][nodeOut] -= h
				layer.costWeightGrad[nodeIn][nodeOut] = differentialChange/h
				
				
			end
		end
		
		
		for bias = 1, #layer.Biases  do
			
			layer.Biases[bias] += h
			local differentialChange = self:Cost(dataset) - orginalCost
			layer.Biases[bias] -= h
			layer.costBiasGrad[bias] = differentialChange/h
			
		end
	end
	
	local value = self:SetAllGradiants(rate)
end

function NeuralNetwork:SetAllGradiants(rate:number)
	for _, layer in self.layer do
		layer:SetGradiants(rate)
	end
end


export type NeuralNetwork = typeof(NeuralNetwork:New())
return NeuralNetwork


