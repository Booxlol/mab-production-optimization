# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================

class Batch:
    def __init__(self, batch_id, ph_opt, temperature_opt):
        self.batch_id = batch_id
        self.ph_opt = ph_opt
        self.temperature_opt = temperature_opt

    def display_info(self):
        print(f"Batch ID: {self.batch_id:4d} | "
              f"Optimal pH: {self.ph_opt:4.1f} | "
              f"Optimal Temperature: {self.temperature_opt:4.1f} C")
