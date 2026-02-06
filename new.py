def count_residents_violating(solver, penalties, assign, days, roles, residents):
    """
    penalties: list of IntVar for this soft constraint
    assign: the assignment dict (d, role, r) -> BoolVar
    days, roles, residents: used to map violations to residents
    Returns: set of resident names that violated
    """
    violated_residents = set()

    for r in residents:
        # If any penalty variable for this resident is 1, count them
        for d in days:
            for role in roles:
                key = (d, role, r)
                if key in assign and any(p.Name().endswith(f"_{d}_{role}_{r}") for p in penalties):
                    if solver.Value(assign[key]) == 1:  # assigned
                        violated_residents.add(r)
                        break  # count resident only once
            else:
                continue
            break

    return violated_residents
