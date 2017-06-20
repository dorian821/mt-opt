def top_up(x):
	if x - round(x,0) > 0:
		x = round(x,0) + 1
	else:
		x = round(x,0)
	return x
