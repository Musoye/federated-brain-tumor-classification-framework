import os
import shutil

saved_path = CONFIG.get("saved_path", "global_model.pth")

def normalize_all_clients(data_root="data/"):
    """
    Normalize each client folder into two classes:
    - no  (no tumor)
    - yes (tumor)

    If a client already contains only "yes" and "no" folders, skip it entirely.
    """

    NO_KEYWORDS = ["notumor", "no", "negative", "no_tumor", "normal"]
    YES_KEYWORDS = ["yes", "tumor", "glioma", "meningioma", "pituitary"]

    # list clients
    clients = [
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ]

    print(f"Found {len(clients)} clients.")

    for client in clients:
        print(f"\nChecking client: {client}")

        subfolders = [
            f for f in os.listdir(client)
            if os.path.isdir(os.path.join(client, f))
        ]

        if set(subfolders) == {"yes", "no"}:
            print("Already normalized → SKIPPING")
            continue

        print(f"🔧 Normalizing: {client}")

        yes_dir = os.path.join(client, "yes")
        no_dir = os.path.join(client, "no")

        os.makedirs(yes_dir, exist_ok=True)
        os.makedirs(no_dir, exist_ok=True)

        # move files from all other folders
        for cls in subfolders:
            cls_path = os.path.join(client, cls)

            # skip new ones
            if cls in ["yes", "no"]:
                continue

            cls_lower = cls.lower()

            # decide target
            if any(k in cls_lower for k in NO_KEYWORDS):
                dest = no_dir
            else:
                dest = yes_dir

            print(f" → Moving {cls_path} → {dest}")

            # move images
            for fname in os.listdir(cls_path):
                src = os.path.join(cls_path, fname)
                if os.path.isfile(src):
                    shutil.move(src, os.path.join(dest, fname))

            # delete old folder
            shutil.rmtree(cls_path)

        print(f"✔ Done: {client}")

    print("\n🎉 All clients normalized into yes/no classes successfully!")

normalize_all_clients(data_root="data/")

out = start_federated(data_root=CONFIG.get("CLIENTS_PATH", "data/"),
                          rounds=CONFIG.get("ROUNDS", 3),
                          clients_per_round=None,
                          eval_every=1,
                          save_path=saved_path,
                          verbose=True)
print("Done. Saved:", out)