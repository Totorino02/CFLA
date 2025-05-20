
def convert_seconds_to_hhmmss(seconds):
    return str(seconds // 3600) + ":" + str((seconds % 3600) // 60) + ":" + str(round(seconds % 60, 2))
