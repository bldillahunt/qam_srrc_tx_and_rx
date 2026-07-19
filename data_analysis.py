def data_analysis(input_data_i, input_data_q):
	max_data_i = []
	max_data_q = []
	std_dev_i = []
	std_dev_q = []
	shift_register_i = np.zeros(4)
	shift_register_q = np.zeros(4)
	
	# Create the shift register containing four samples
	for i in range(0, len(input_data_i)):
		shift_register_i[0] = input_data_i[i]
		shift_register_q[0] = input_data_q[i]
		
		for j in range(1, 4):
			shift_register_i[j] = shift_register_i[j-1]
			shift_register_q[j] = shift_register_q[j-1]
	
		std_dev_i.append(np.std(shift_register_i))
		std_dev_q.append(np.std(shift_register_q))
		max_data_i.append(abs(max(shift_register_i)))
		max_data_q.append(abs(max(shift_register_q)))

	return std_dev_i, std_dev_q, max_data_i, max_data_q
	