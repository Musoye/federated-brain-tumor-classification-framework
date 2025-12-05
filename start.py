out = start_federated(data_root=CONFIG.get("CLIENTS_PATH", "data/"),
                          rounds=CONFIG.get("ROUNDS", 3),
                          clients_per_round=None,
                          eval_every=1,
                          save_path="global_model.pth",
                          verbose=True)
print("Done. Saved:", out)