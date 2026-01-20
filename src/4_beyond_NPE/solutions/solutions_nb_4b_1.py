"""Solution for SNPE multi-round inference exercise"""

# Initialize
trainer = SNPE(prior)
proposal = prior  # Start sampling from the prior
posteriors = []   # Store posteriors from each round

for round_idx in range(NUM_ROUNDS):
    print(f"\n=== Round {round_idx + 1} ===")
    
    # TODO Step 1: Generate training data by sampling from the proposal
    # Use simulate_for_sbi(simulator, proposal, NUM_SIMS_PER_ROUND, num_workers=NUM_WORKERS)
    print(f"Simulating {NUM_SIMS_PER_ROUND} samples...")
    theta, x = simulate_for_sbi(simulator, proposal, NUM_SIMS_PER_ROUND, num_workers=NUM_WORKERS)
    
    # TODO Step 2: Append simulations to trainer
    # Important: pass proposal=proposal so SNPE knows where samples came from
    trainer.append_simulations(theta, x, proposal=proposal)
    
    # TODO Step 3: Train
    print("Training...")
    trainer.train(show_train_summary=True)
    
    # TODO Step 4: Build posterior and store it
    posterior = trainer.build_posterior()
    posteriors.append(posterior)
    
    # TODO Step 5: Update proposal for next round
    # Use posterior.set_default_x(x_o) to condition on our observation
    proposal = posterior.set_default_x(x_o)

print("\n=== Done ===")