from scipy.stats import poisson

avg_arrival_rate = 5  
real_time_calls = 8  
call_prob = poisson.pmf(real_time_calls, avg_arrival_rate)

print(f"Probability of receiving {real_time_calls} calls in a minute: {call_prob:.4f}")

if real_time_calls > 6:
    print("High traffic! Consider adding more staff.")
else:
    print("Traffic is normal.")
