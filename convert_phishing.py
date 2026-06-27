import mailbox
import os
import glob

def convert_mbox(mbox_path, out_dir):
    """
    Extract all emails from an mbox file into individual .eml files.
    Uses binary mode to avoid charset issues.
    """
    mbox = mailbox.mbox(mbox_path)
    base = os.path.splitext(os.path.basename(mbox_path))[0]  # e.g., phishing-2018
    count = 0
    skipped = 0
    for msg in mbox:
        try:
            # Create a unique filename
            eml_name = f"{base}_{count:05d}.eml"
            eml_path = os.path.join(out_dir, eml_name)
            # Write the exact raw bytes of the email
            with open(eml_path, 'wb') as f:
                f.write(msg.as_bytes())
            count += 1
            if count % 500 == 0:
                print(f"  {count} emails extracted...")
        except Exception as e:
            # Skip any completely unparseable messages
            skipped += 1
            if skipped <= 5:  # Print only first few errors to avoid spam
                print(f"  Skipping one malformed message: {e}")
    print(f"  Done. {count} emails written, {skipped} skipped.")
    return count

if __name__ == '__main__':
    phishing_dir = r"D:\youssef hefzy\College\Graduation project\current work\EmailHeaderAnomalyDetection_Extended\data\raw\phishing\phishing"
    out_dir = phishing_dir

    # Find all .txt files that are actually mbox files
    mbox_files = sorted(glob.glob(os.path.join(phishing_dir, "phishing-*.txt")))
    if not mbox_files:
        print("No mbox files found in", phishing_dir)
        print("Make sure phishing-2017.txt, phishing-2018.txt, etc. are there.")
    else:
        print(f"Found {len(mbox_files)} mbox files.")
        for mbox_path in mbox_files:
            print(f"Processing {os.path.basename(mbox_path)} ...")
            convert_mbox(mbox_path, out_dir)

        # Delete the original .txt files so they aren't processed by the pipeline
        print("\nExtraction complete. Deleting original .txt files...")
        for mbox_path in mbox_files:
            os.remove(mbox_path)
            print(f"  Deleted {os.path.basename(mbox_path)}")
        print("Done. Now you can run the main pipeline.")