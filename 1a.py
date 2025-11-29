def calculate_probability(favorable_outcomes, total_outcomes):
    if total_outcomes == 0:
        return 0  
    return favorable_outcomes / total_outcomes

total_outcomes = 6

favorable_outcomes = 1

probability = calculate_probability(favorable_outcomes, total_outcomes)

print(f"The probability of rolling a 3 is: {probability:.2f}")
