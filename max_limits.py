def calculate_max_limits(residents, nf_residents, wr_residents, night_counts, resident_max_limit, nf_max_limit ):
    """
    Calculate maximum shifts, points, and weekend limits for each resident.
    """
    max_shifts, max_points, weekend_limits = {}, {}, {}

    for resident in residents:      # WIP dynamic max
        if resident in nf_residents:        # night floaters
            shifts, points, weekend = (nf_max_limit)
        else: 
            shifts, points, weekend = (resident_max_limit)
            
        # Define Max Shifts, Max Points, Max Weekends
        max_shifts[resident] = shifts
        max_points[resident] = points
        weekend_limits[resident] = weekend

    return max_shifts, max_points, weekend_limits