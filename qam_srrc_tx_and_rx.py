from re import A
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import matplotlib.pyplot as plt
from scipy import signal
import commpy
from commpy.filters import rrcosfilter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.signal import firwin, lfilter

INTEGER_BITS = 32
INTEGER_SCALE = 2**INTEGER_BITS
DATA_SIZE = 2048
#SAMPLES_PER_SYMBOL = 8
MODULATED_BITS = 4
#UPSAMPLE_DATA_SIZE = SAMPLES_PER_SYMBOL*DATA_SIZE*(INTEGER_BITS/MODULATED_BITS)
RF_SAMPLE_RATE = 5e+9
#DATA_SAMPLE_RATE = RF_SAMPLE_RATE/SAMPLES_PER_SYMBOL

def prbs_generator(seed_value):
	mask = 0xFFFFFFFF
	# Polynomial = x^32 + x^22 + x^2 + x^1 + 1
	lfsr_bit	= (seed_value ^ (seed_value >> 10) ^ (seed_value >> 30) ^ (seed_value >> 31)) & 0x1
	lfsr_data	= ((seed_value << 1) | (lfsr_bit)) & mask
#	print('bit = ', lfsr_bit, 'data = ', hex(lfsr_data))
#	char = sys.stdin.read(1)
	return lfsr_data

def qam4_modulation(bits):
	symbols = []
    
	for i in range(0, len(bits), MODULATED_BITS):
		if bits[i] == 0 and bits[i+1] == 0:
			symbols.append(-1 - 1j)
		elif bits[i] == 0 and bits[i+1] == 1:
			symbols.append(-1 + 1j)
		elif bits[i] == 1 and bits[i+1] == 0:
			symbols.append(1 - 1j)
		else:
			symbols.append(1 + 1j)

#	print('symbols = ', symbols)
	return symbols

def qam16_modulation(bits):
	symbols = []
	current_index = 0
	for i in range(0, len(bits), MODULATED_BITS):
		nibble_data = [bits[i+3], bits[i+2], bits[i+1], bits[i]]

		if (nibble_data == [0, 0, 0, 0]):
			symbols.append(-1-1j)	# 0
		elif (nibble_data == [0, 0, 0, 1]):
			symbols.append(-1-3j)	# 1
		elif (nibble_data == [0, 0, 1, 0]):
			symbols.append(-1+1j)	# 2
		elif (nibble_data == [0, 0, 1, 1]):
			symbols.append(-1+3j)	# 3
		elif (nibble_data == [0, 1, 0, 0]):
			symbols.append(-3-1j)	# 4
		elif (nibble_data == [0, 1, 0, 1]):
			symbols.append(-3-3j)	# 5
		elif (nibble_data == [0, 1, 1, 0]):
			symbols.append(-3+1j)	# 6
		elif (nibble_data == [0, 1, 1, 1]):
			symbols.append(-3+3j)	# 7
		elif (nibble_data == [1, 0, 0, 0]):
			symbols.append(1-1j)	# 8
		elif (nibble_data == [1, 0, 0, 1]):
			symbols.append(1-3j)	# 9
		elif (nibble_data == [1, 0, 1, 0]):
			symbols.append(1+1j)	# 10
		elif (nibble_data == [1, 0, 1, 1]):
			symbols.append(1+3j)	# 11
		elif (nibble_data == [1, 1, 0, 0]):
			symbols.append(3-1j)	# 12
		elif (nibble_data == [1, 1, 0, 1]):
			symbols.append(3-3j)	# 13
		elif (nibble_data == [1, 1, 1, 0]):
			symbols.append(3+1j)	# 14
		elif (nibble_data == [1, 1, 1, 1]):
			symbols.append(3+3j)	# 15
#		print('i = ', i, 'data = ', symbols[current_index])
		current_index = current_index + 1
#		sys.stdin.read(1)
	return symbols

def print_data_to_file(input_data, data_file):
	with open(data_file, 'w') as f:
		for item in input_data:
			f.write(f"{item}\n") # Write each item followed by a newline

def find_likely_coordinates(i_modulated_data, q_modulated_data):
	constellation_data = [(-1-1j), (-1-3j), (-1+1j), (-1+3j), (-3-1j), (-3-3j), (-3+1j), (-3+3j),	(1-1j), (1-3j),	(1+1j),	(1+3j),	(3-1j),	(3-3j),	(3+1j),	(3+3j)]
	phase_offsets = []
	magnitude_offsets = []
	binary_data_output = []

	# Loop through every data sample
	for i in range(0, len(i_modulated_data)):
		error_vector = []
		error_magnitude = []
		minimum_error = 2**31

		# Loop through the constellation
		for j in range(0, 2**MODULATED_BITS):
			error_vector.append((i_modulated_data[i]-constellation_data[j].real) + 1j*(q_modulated_data[i]-constellation_data[j].imag))
			error_magnitude.append(math.sqrt((i_modulated_data[i]-constellation_data[j].real)**2 + (q_modulated_data[i]-constellation_data[j].imag)**2))

			if (error_magnitude[j] < minimum_error):
				minimum_error = error_magnitude[j]
				symbol_index = j

		binary_data_output.append(symbol_index)

	return binary_data_output

def grid_search_mapping(i_coordinates, q_coordinates):
	binary_data_output = []
	
	for i in range(0, len(i_coordinates)):
		# 0
		if ((i_coordinates[i] < 0) and (i_coordinates[i] > -2) and (q_coordinates[i] < 0) and (q_coordinates[i] > -2)):
			binary_data_output.append(0)
		# 1
		elif ((i_coordinates[i] < 0) and (i_coordinates[i] > -2) and (q_coordinates[i] < 0) and (q_coordinates[i] < -2)):
			binary_data_output.append(1)
		# 2
		elif ((i_coordinates[i] < 0) and (i_coordinates[i] > -2) and (q_coordinates[i] > 0) and (q_coordinates[i] < 2)):
			binary_data_output.append(2)
		# 3
		elif ((i_coordinates[i] < 0) and (i_coordinates[i] > -2) and (q_coordinates[i] > 0) and (q_coordinates[i] > 2)):
			binary_data_output.append(3)
		# 4
		elif ((i_coordinates[i] < 0) and (i_coordinates[i] < -2) and (q_coordinates[i] < 0) and (q_coordinates[i] > -2)):
			binary_data_output.append(4)
		# 5
		elif ((i_coordinates[i] < 0) and (i_coordinates[i] < -2) and (q_coordinates[i] < 0) and (q_coordinates[i] < -2)):
			binary_data_output.append(5)
		# 6
		elif ((i_coordinates[i] < 0) and (i_coordinates[i] < -2) and (q_coordinates[i] > 0) and (q_coordinates[i] < 2)):
			binary_data_output.append(6)
		# 7
		elif ((i_coordinates[i] < 0) and (i_coordinates[i] < -2) and (q_coordinates[i] > 0) and (q_coordinates[i] > 2)):
			binary_data_output.append(7)
		# 8
		elif ((i_coordinates[i] > 0) and (i_coordinates[i] < 2) and (q_coordinates[i] < 0) and (q_coordinates[i] > -2)):
			binary_data_output.append(8)
		# 9
		elif ((i_coordinates[i] > 0) and (i_coordinates[i] < 2) and (q_coordinates[i] < 0) and (q_coordinates[i] < -2)):
			binary_data_output.append(9)
		# 10
		elif ((i_coordinates[i] > 0) and (i_coordinates[i] < 2) and (q_coordinates[i] > 0) and (q_coordinates[i] < 2)):
			binary_data_output.append(10)
		# 11
		elif ((i_coordinates[i] > 0) and (i_coordinates[i] < 2) and (q_coordinates[i] > 0) and (q_coordinates[i] > 2)):
			binary_data_output.append(11)
		# 12
		elif ((i_coordinates[i] > 0) and (i_coordinates[i] > 2) and (q_coordinates[i] < 0) and (q_coordinates[i] > -2)):
			binary_data_output.append(12)
		# 13
		elif ((i_coordinates[i] > 0) and (i_coordinates[i] > 2) and (q_coordinates[i] < 0) and (q_coordinates[i] < -2)):
			binary_data_output.append(13)
		# 14
		elif ((i_coordinates[i] > 0) and (i_coordinates[i] > 2) and (q_coordinates[i] > 0) and (q_coordinates[i] < 2)):
			binary_data_output.append(14)
		# 15
		elif ((i_coordinates[i] > 0) and (i_coordinates[i] > 2) and (q_coordinates[i] > 0) and (q_coordinates[i] > 2)):
			binary_data_output.append(15)
		else:
			binary_data_output.append(0)
			print('i = ', i, 'I coordinate = ', i_coordinates[i], 'Q coordinate = ', q_coordinates[i])
#			sys.stdin.read(1)
			
	return binary_data_output

def print_hex_to_file(input_data, data_file):
	with open(data_file, 'w') as f:
		for item in input_data:
			f.write(f"{hex(int(item))}\n") # Write each item followed by a newline

def iq_time_domain_plot(signal_time, sample_rate, plot_title:str, i_input, q_input):
	signal_output_time = np.arange(0, len(i_input)/sample_rate, 1/sample_rate)
	plt.plot(signal_output_time, i_input, color='blue')
	plt.plot(signal_output_time, q_input, color='red')
	plt.xlabel("Time (s)")
	plt.ylabel("Amplitude")
	plt.title(plot_title)
	plt.grid(True)
	plt.show()

def time_domain_plot(signal_time, sample_rate, plot_title:str, signal_input):
	signal_output_time = np.arange(0, signal_time, 1/sample_rate)
	plt.plot(signal_output_time, signal_input, color='blue')
	plt.xlabel("Time (s)")
	plt.ylabel("Amplitude")
	plt.title(plot_title)
	plt.grid(True)
	plt.show()

def plot_unit_circle(symbol_data, plot_title):
	x_coords = [c.real for c in symbol_data]
	y_coords = [c.imag for c in symbol_data]

	plt.figure(figsize=(6, 6))
	plt.scatter(x_coords, y_coords, color='red', marker='o')
	plt.xlabel("Real Part")
	plt.ylabel("Imaginary Part")
	plt.title(plot_title)
	plt.grid(True)
	plt.axhline(0, color='black',linewidth=0.5)
	plt.axvline(0, color='black',linewidth=0.5)
	plt.show()

def fft_generate_and_plot(input_data, sample_count, sample_rate, plot_title:str):
	signal_spectrum = np.fft.fft(input_data)
	signal_frequencies = np.fft.fftfreq(sample_count, d=1/sample_rate)

	spectrum_shifted = np.fft.fftshift(signal_spectrum)
	frequencies_shifted = np.fft.fftshift(signal_frequencies)

	plt.figure(figsize=(10, 5))
	plt.plot(frequencies_shifted, np.abs(spectrum_shifted))
	plt.title(plot_title)
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Magnitude')
	plt.grid(True)
	plt.show()

def complex_rescaling(complex_data_in, minimum_value, maximum_value):
	i_data_in = []
	q_data_in = []

	for i in range(0, len(complex_data_in)):
		i_data_in.append(complex_data_in[i].real)
		q_data_in.append(complex_data_in[i].imag)
		print('i_data_in = ', i_data_in[i], 'q_data_in = ', q_data_in[i])
		char = sys.stdin.read(1)	
    
	# New value = ((old value - old min)/(old max - old min))x(new max - new min) + new min
	data_input_min_i = np.min(i_data_in)
	data_input_max_i = np.max(i_data_in)
	data_input_min_q = np.min(q_data_in)
	data_input_max_q = np.max(q_data_in)
	scale_min = minimum_value
	scale_max = maximum_value

	print('min I = ', data_input_min_i, 'min Q = ', data_input_min_q, 'max I = ', data_input_max_i, 'max Q = ', data_input_max_q)
	char = sys.stdin.read(1)
	
	scaled_data_array_i = []
	scaled_data_array_q = []

	for i in range(0, len(i_data_in)):
		scaled_data_array_i.append(((i_data_in[i]-data_input_min_i)/(data_input_max_i-data_input_min_i))*(scale_max - scale_min) + scale_min)
		scaled_data_array_q.append(((q_data_in[i]-data_input_min_q)/(data_input_max_q-data_input_min_q))*(scale_max - scale_min) + scale_min)

	scaled_data_array = []

	for i in range(0, len(scaled_data_array_i)):
		scaled_data_array.append(scaled_data_array_i[i] + 1j*scaled_data_array_q[i])
	
	return scaled_data_array

def scalar_rescaling(scalar_data_in, minimum_value, maximum_value):
	# New value = ((old value - old min)/(old max - old min))x(new max - new min) + new min
	data_input_min = np.min(scalar_data_in)
	data_input_max = np.max(scalar_data_in)
	scaled_data_array = []
	print('min = ', data_input_min, 'max = ', data_input_max)
	
	for i in range(0, len(scalar_data_in)):
		scaled_data_array.append(((scalar_data_in[i]-data_input_min)/(data_input_max-data_input_min))*(maximum_value - minimum_value) + minimum_value)
#		print('i = ', i, 'data in = ', scalar_data_in[i], 'scaled data = ', scaled_data_array[i])

#	char = sys.stdin.read(1)

	return scaled_data_array

# Note: The actual data throughput will be RF_SAMPLE_RATE/SAMPLES_PER_SYMBOL
# The cutoff frequency should be greater than the symbol rate but less than the carrier

enable_filters = input("Enable filters (Y = yes, N = no )")
srrc_tap_count = input("Enter number of taps: ")
#cutoff_frequency = input("Enter cutoff frequency: ")
SAMPLES_PER_SYMBOL = int(input("Enter samples per symbol: "))
rolloff_factor = float(input("Enter the roll-off factor: "))
phase_offset = int(input("Enter the number of clock shifts: "))

DATA_SIZE = 2048
DURATION = (DATA_SIZE*int(INTEGER_BITS/MODULATED_BITS)*SAMPLES_PER_SYMBOL)/RF_SAMPLE_RATE

#----------<<<<<<<<<< STEP 1 >>>>>>>>>>----------

# Create the random data array
# PRBS data generator

PREAMBLE = 0xF0F0F0F0
TRAINING_PATTERN = 0xDEADBEEF
seed_value = 0xFFFFFFFF
random_data = []

random_data.append(PREAMBLE)
random_data.append(TRAINING_PATTERN)

for i in range(0, DATA_SIZE-2):
	current_pattern = int(prbs_generator(seed_value))
	seed_value = current_pattern
	random_data.append(seed_value)

print_hex_to_file(random_data, 'original_data_set.txt')

#----------<<<<<<<<<< STEP 2 >>>>>>>>>>----------

# Convert each bit to binary
tx_clocks_per_sample_time = DATA_SIZE*(INTEGER_BITS/MODULATED_BITS)*SAMPLES_PER_SYMBOL
rows = int(tx_clocks_per_sample_time)
cols = INTEGER_BITS
default_value = 0
binary_data = [[default_value for _ in range(cols)] for _ in range(rows)]
symbols_array = []	#[[default_value for _ in range(int(cols/2))] for _ in range(rows)]
symbols_array_complex = []

for i in range(0, DATA_SIZE):
	shift_register = random_data[i]

	# Convert the random data to binary (32 bits)
	for j in range(0, INTEGER_BITS):
		binary_data[i][j] = shift_register & 0x00000001
		shift_register = shift_register >> 1
	
	# Convert the 32 bit value to complex numbers (1+1*j, -1+1*j, -1-1*j, 1-1*j)
	# This reduces the array to 16 complex numbers
	symbols_array.append(qam16_modulation(binary_data[i]))
	symbol_register = [complex(s) for s in symbols_array[i]]
	symbols_array_complex.append(symbol_register)

#----------<<<<<<<<<< STEP 3 >>>>>>>>>>----------

# Flatten the data
symbol_data_flat = []

for i in range(0, DATA_SIZE):
	for j in range(0, int(INTEGER_BITS/MODULATED_BITS)):
		symbol_data_flat.append(symbols_array_complex[i][j])

plot_unit_circle(symbol_data_flat, 'RANDOM DATA BEFORE INTERPOLATION')

#----------<<<<<<<<<< STEP 4 >>>>>>>>>>----------

# Interpolate the symbol data to account for multiple samples per symbol
symbol_data_up = np.zeros(len(symbol_data_flat) * SAMPLES_PER_SYMBOL, dtype=complex)
symbol_data_up[::SAMPLES_PER_SYMBOL] = symbol_data_flat

print_data_to_file(symbol_data_up, 'interpolated_random_data.txt')
plot_unit_circle(symbol_data_up, 'RANDOM DATA AFTER INTERPOLATION')

#----------<<<<<<<<<< STEP 5 >>>>>>>>>>----------

# Pass the data through a srrc filter

if (enable_filters == 'Y'):
	TX_N = int(srrc_tap_count)							# Filter length (taps)
	tx_alpha = rolloff_factor							# Roll-off factor
	tx_Ts = SAMPLES_PER_SYMBOL/RF_SAMPLE_RATE			# Symbol duration
	tx_Fs = RF_SAMPLE_RATE				# Sample rate
#	tx_cutoff_frequency = int(cutoff_frequency)
#	tx_srrc_taps = firwin(TX_N, tx_cutoff_frequency, fs=RF_SAMPLE_RATE, window='hamming')
	tx_t, tx_srrc_taps = rrcosfilter(TX_N, tx_alpha, tx_Ts, tx_Fs)
	
	print_data_to_file(tx_srrc_taps, 'tx_srrc_taps.txt')

	# Normalize the coefficients
	tx_srrc_taps_normalized = scalar_rescaling(tx_srrc_taps, 0, 1)
	#tx_srrc_taps_normalized = tx_srrc_taps
	
	print_data_to_file(tx_srrc_taps_normalized, 'tx_srrc_taps_normalized.txt')

	tx_symbol_data_i = []	
	tx_symbol_data_q = []

	# Separate the I and Q data
	for i in range(0, DATA_SIZE*int(INTEGER_BITS/MODULATED_BITS)*SAMPLES_PER_SYMBOL):
		tx_symbol_data_i.append(symbol_data_up[i].real)
		tx_symbol_data_q.append(symbol_data_up[i].imag)

	# Pass the symbol data through the filter
#	symbol_data_filtered_i = lfilter(tx_srrc_taps_normalized, [1.0], tx_symbol_data_i)
#	symbol_data_filtered_q = lfilter(tx_srrc_taps_normalized, [1.0], tx_symbol_data_q)
	symbol_data_filtered_i = np.convolve(tx_symbol_data_i, tx_srrc_taps_normalized, mode='full')
	symbol_data_filtered_q = np.convolve(tx_symbol_data_q, tx_srrc_taps_normalized, mode='full')

	print_data_to_file(symbol_data_filtered_i, 'transmit_symbol_data_filtered_i.txt')
	print_data_to_file(symbol_data_filtered_q, 'transmit_symbol_data_filtered_q.txt')

	# Eliminate the group delay
 
	tx_reduced = []

#	for i in range(TX_N-1, DATA_SIZE*int(INTEGER_BITS/MODULATED_BITS)*SAMPLES_PER_SYMBOL+TX_N-1):
	for i in range(TX_N-1, len(symbol_data_filtered_i)):
		tx_reduced.append(symbol_data_filtered_i[i] + 1j*symbol_data_filtered_q[i])
	
	plot_unit_circle(tx_reduced, 'TRANSMIT DATA AFTER REDUCTION')
	print_data_to_file(tx_reduced, 'tx_reduced.txt')
else:
	tx_symbol_data_i = []	
	tx_symbol_data_q = []

	for i in range(0, DATA_SIZE*int(INTEGER_BITS/MODULATED_BITS)*SAMPLES_PER_SYMBOL):
		tx_symbol_data_i.append(symbol_data_up[i].real)
		tx_symbol_data_q.append(symbol_data_up[i].imag)

	symbol_data_filtered_i = tx_symbol_data_i
	symbol_data_filtered_q = tx_symbol_data_q

	tx_reduced = []

	for i in range(0, DATA_SIZE*int(INTEGER_BITS/MODULATED_BITS)*SAMPLES_PER_SYMBOL):
		tx_reduced.append(symbol_data_filtered_i[i] + 1j*symbol_data_filtered_q[i])

	print_data_to_file(tx_reduced, 'tx_reduced.txt')
	print('length of tx_reduced = ', len(tx_reduced))

#----------<<<<<<<<<< STEP 6 >>>>>>>>>>----------

# Create the other half of the srrc filter
if (enable_filters == 'Y'):
	RX_N = int(srrc_tap_count)							# Filter length (taps)
	rx_alpha = rolloff_factor							# Roll-off factor
	rx_Ts = SAMPLES_PER_SYMBOL/RF_SAMPLE_RATE		# Symbol duration
	rx_Fs = RF_SAMPLE_RATE				# Sampling rate (4 samples per symbol)
#	rx_cutoff_frequency = int(cutoff_frequency)
	# Generate filter coefficients and time vector
#	rx_srrc_taps = firwin(TX_N, rx_cutoff_frequency, fs=RF_SAMPLE_RATE, window='hamming')
	rx_t, rx_srrc_taps = rrcosfilter(RX_N, rx_alpha, rx_Ts, rx_Fs)

	# Normalize the coefficients
	rx_srrc_taps_normalized = scalar_rescaling(rx_srrc_taps, 0, 1)
	#rx_srrc_taps_normalized = rx_srrc_taps
	
	print_data_to_file(rx_srrc_taps_normalized, 'rx_srrc_taps_normalized.txt')

	rx_symbol_data_i = []	
	rx_symbol_data_q = []
	print_data_to_file(rx_srrc_taps, 'rx_srrc_taps.txt')

	for i in range(0, DATA_SIZE*int(INTEGER_BITS/MODULATED_BITS)*SAMPLES_PER_SYMBOL-(RX_N-1)):
		rx_symbol_data_i.append(tx_reduced[i].real)
		rx_symbol_data_q.append(tx_reduced[i].imag)

#	rx_filtered_signal_i = lfilter(rx_srrc_taps_normalized, [1.0], rx_symbol_data_i)
#	rx_filtered_signal_q = lfilter(rx_srrc_taps_normalized, [1.0], rx_symbol_data_q)
	rx_filtered_signal_i = np.convolve(rx_symbol_data_i, rx_srrc_taps_normalized, mode='full')
	rx_filtered_signal_q = np.convolve(rx_symbol_data_q, rx_srrc_taps_normalized, mode='full')

	# Remove the group delay data

	rx_reduced = []
	rx_reduced_i = []
	rx_reduced_q = []

#	for i in range(RX_N-1, DATA_SIZE*int(INTEGER_BITS/MODULATED_BITS)*SAMPLES_PER_SYMBOL+RX_N-1):
	for i in range((RX_N-1)+phase_offset, len(rx_filtered_signal_i)):
		rx_reduced.append(rx_filtered_signal_i[i] + 1j*rx_filtered_signal_q[i])
		rx_reduced_i.append(rx_filtered_signal_i[i])
		rx_reduced_q.append(rx_filtered_signal_q[i])

	plot_unit_circle(rx_reduced, 'RECEIVE DATA FILTERED AND REDUCED')
	print_data_to_file(rx_filtered_signal_i, 'receive_data_filtered_i.txt')
	print_data_to_file(rx_filtered_signal_q, 'receive_data_filtered_q.txt')
	iq_time_domain_plot(DURATION, RF_SAMPLE_RATE, 'Reduced Data Plot', rx_reduced_i, rx_reduced_q)
else:
	rx_reduced = tx_reduced

#----------<<<<<<<<<< STEP 7 >>>>>>>>>>----------

rx_decimated = signal.decimate(rx_reduced, SAMPLES_PER_SYMBOL, ftype='fir', zero_phase=True)
#rx_decimated = rx_reduced
print_data_to_file(rx_decimated, 'reduced_filtered_data.txt')
print('length of rx_decimated = ', len(rx_decimated))
phase_array = []
magnitude_array = []
rx_decimated_i = []
rx_decimated_q = []

for i in range(0, len(rx_decimated)):
	magnitude_array.append(math.sqrt(rx_decimated[i].real**2 + rx_decimated[i].imag**2))
	phase_array.append(np.angle(rx_decimated[i]))
	rx_decimated_i.append(rx_decimated[i].real)
	rx_decimated_q.append(rx_decimated[i].imag)

print_data_to_file(phase_array, 'phase_array.txt')
plot_unit_circle(rx_decimated, 'DECIMATED RECEIVE DATA')
iq_time_domain_plot(len(rx_decimated_i)/RF_SAMPLE_RATE, RF_SAMPLE_RATE, 'Receiver Decimated Data', rx_decimated_i, rx_decimated_q)
print_data_to_file(rx_decimated_i, 'rx_decimated_i.txt')
print_data_to_file(rx_decimated_q, 'rx_decimated_q.txt')

# Rescaling equation:
# New value = ((old value - old min)/(old max - old min))x(new max - new min) + new min
rx_decimated_min_i = np.min(rx_decimated_i)
rx_decimated_max_i = np.max(rx_decimated_i)
rx_scale_min_i = -math.sqrt(3**2 + 3**2)
rx_scale_max_i = math.sqrt(3**2 + 3**2)

rx_decimated_min_q = np.min(rx_decimated_q)
rx_decimated_max_q = np.max(rx_decimated_q)
rx_scale_min_q = -math.sqrt(3**2 + 3**2)
rx_scale_max_q = math.sqrt(3**2 + 3**2)

rx_decimated_scaled_i = []
rx_decimated_scaled_q = []

for i in range(0, len(rx_decimated_i)):
	rx_decimated_scaled_i.append(((rx_decimated_i[i]-rx_decimated_min_i)/(rx_decimated_max_i-rx_decimated_min_i))*(rx_scale_max_i - rx_scale_min_i) + rx_scale_min_i)

for i in range(0, len(rx_decimated_q)):
	rx_decimated_scaled_q.append(((rx_decimated_q[i]-rx_decimated_min_q)/(rx_decimated_max_q-rx_decimated_min_q))*(rx_scale_max_q - rx_scale_min_q) + rx_scale_min_q)

rx_decimated_output = [complex(i, q) for i, q in zip(rx_decimated_scaled_i, rx_decimated_scaled_q)]
plot_unit_circle(rx_decimated_output, 'RESCALED RECEIVE DATA')
iq_time_domain_plot(len(rx_decimated_scaled_i)/RF_SAMPLE_RATE, RF_SAMPLE_RATE, 'Receiver Rescaled Data', rx_decimated_scaled_i, rx_decimated_scaled_q)

print_data_to_file(rx_decimated_scaled_i, 'rx_decimated_scaled_i.txt')
print_data_to_file(rx_decimated_scaled_q, 'rx_decimated_scaled_q.txt')

print('length of rx_decimated_scaled_i = ', len(rx_decimated_scaled_i))
print('length of rx_decimated_scaled_q = ', len(rx_decimated_scaled_q))

rx_binary_data = []
rx_binary_data = find_likely_coordinates(rx_decimated_scaled_i, rx_decimated_scaled_q)
#rx_binary_data = grid_search_mapping(rx_decimated_scaled_i, rx_decimated_scaled_q)
print_data_to_file(rx_binary_data, 'rx_binary_data.txt')

print('length of rx_binary_data = ', len(rx_binary_data))

# Convert the binary array to 32 bit values

# First, locate training pattern in the binary data
training_pattern_found = False
rx_data_index = 0
rx_shift_register = 0
start_index = 0

while ((training_pattern_found == False) and (rx_data_index < len(rx_binary_data))):
	rx_shift_register = (rx_shift_register >> MODULATED_BITS) | (rx_binary_data[rx_data_index] << (INTEGER_BITS-MODULATED_BITS)) 
	print('index = ', rx_data_index, 'shift_register = ', hex(rx_shift_register), 'binary data = ', rx_binary_data[rx_data_index])
	rx_data_index = rx_data_index + 1
#	char = sys.stdin.read(1)

	if (rx_shift_register == TRAINING_PATTERN):
		start_index = rx_data_index
		training_pattern_found = True
		print('training pattern found')
	elif (rx_data_index == len(rx_binary_data)):
		print('training pattern not found')

# Put the nibbles together to get 32 bit words
rx_data_word = []
current_index = start_index

# Start reading the binary array at the current start index
for i in range(0, int(len(rx_binary_data)/(int(INTEGER_BITS/MODULATED_BITS)))):	# Loop through the entire array one at a time
	binary_shift_register = 0

	# Grab the data based on the modulation scheme and on the word size
	for j in range(0, int(INTEGER_BITS/MODULATED_BITS)):
		if ((current_index+j) < len(rx_binary_data)):
			binary_shift_register = (binary_shift_register >> MODULATED_BITS) | (rx_binary_data[current_index+j] << (INTEGER_BITS-MODULATED_BITS))
	
	current_index = current_index + int(INTEGER_BITS/MODULATED_BITS);
	rx_data_word.append(binary_shift_register)

print_hex_to_file(rx_data_word, 'rx_data_word.txt')

# Compare the input to the output
error_count = 0

for i in range(0, len(rx_data_word)-2):
	if (rx_data_word[i] != random_data[i+2]):
		print('Data mismatch: i = ', i, 'output data = ', hex(rx_data_word[i]), 'input data = ', hex(random_data[i+2]))
		error_count = error_count + 1

print('Error count = ', error_count)

