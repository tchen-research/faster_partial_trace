def binary_search_recursive(func, start, end, iterations, tolerance, turning_points, func_start=None, func_end=None):
    '''
    Uses recursive binary search method to find where the data set flips between "high" and "low" data.
    Input
    -----
    func: function used to obtain datapoints
    start: minimum x-value in search space
    end: maximum x-value in search space
    iterations: number of times to estimate the turning point
    tolerance: maximum distance between the y-value of the start point and the y-value of the guess point to be considered equal
    turning_points: list of all currently found turning points
    
    Output
    -----
    turning_points becomes populated with pairs of x-values that correspond to a change in y values
    '''
       
    # Continue examining the midpoint of the search space
    while(iterations > 0):

        # Define the midpoint of the interval
        guess = (start + end)/2
        func_guess = func(guess)

        if func_start is None:
            func_start = func(start)
        if func_end is None:
            func_end = func(end)
        
        print(f'{start},{end},{func_start:1.3e},{func_end:1.3e},{func_guess:1.3e},{iterations}')

        if(abs(func_start - func_end) <= tolerance):
            return 
            
        # Reduce the search space
        if(abs(func_guess - func_start) <= tolerance):
            # If the von Neumann Entropy at the midpoint is approximately equal to the von Neumann Entropy at the start of
            # the interval, move the start of the interval to the midpoint
            start = guess
            func_start = func_guess
        elif(abs(func_guess - func_end) <= tolerance):
            # If the von Neumann Entropy at the midpoint is approximately equal to the von Neumann Entropy at the end of
            # the interval, move the end of the interval to the midpoint
            end = guess
            func_end = func_guess
        else:
            # If the von Neumann Entropy at the midpoint is not equal to the von Neumann Entropy at the start or end of
            # the interval, call the function again to first search the first half of the interval and then search the
            # second half of the interval
            binary_search_recursive(func, start, guess, iterations-1, tolerance, turning_points, func_start=func_start, func_end = func_guess)
            start = guess
            func_start = func_guess

        # Reduce the number times to continue examining the search space by 1
        iterations = iterations - 1

    guess = (start + end)/2
    # Append the estimated turning point to the list
    turning_points.append(guess)
    
    # (Unnecessary) Return the estimated turning point
    # return guess