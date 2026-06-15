class Schedule:
    """
    Represents a sparse schedule defined by index-based, periodic, and initial-only actions.

    Attributes:
        total_steps (int): total number of cycles (0 to total_steps-1).
        actions (list of [code, spec]):
            spec == 0: run only at step 0;
            spec > 0: run at explicit indices from index_lists[spec];
            spec < 0: run every |spec| steps.
        index_lists (dict): maps positive spec ints to explicit step lists.
    """
    def __init__(self, total_steps, actions, index_lists):
        self.total_steps = total_steps
        self.actions = actions
        self.index_lists = index_lists

    def get_actions(self, step):
        result = []
        for code, spec in self.actions:
            if spec == 0:
                if step == 0:
                    result.append(code)
            elif spec > 0:
                if spec in self.index_lists and step in self.index_lists[spec]:
                    result.append(code)
            else:  # spec < 0
                interval = -spec
                if step % interval == 0:
                    result.append(code)
            
        return result

    def iter_schedule(self):
        for step in range(self.total_steps):
            yield step, self.get_actions(step)

    def display(self, enumerate_steps=False, show_lists=False):
        if enumerate_steps:
            for step, acts in self.iter_schedule():
                line = f"Step {step} "
                line += ": " + ', '.join(acts) if acts else ": (no actions)"
                print(line)
        else:
            print(f"Schedule {self.total_steps} steps")
            # Group codes by their spec
            grouped = {}
            for code, spec in self.actions:
                grouped.setdefault(spec, []).append(code)
        
            # Print each group on one line
            for spec, codes in grouped.items():
                if spec == 0:
                    meaning = "step 0"
                elif spec > 0:
                    meaning = f"list {spec}"
                elif spec == -1:
                    meaning = "every step"
                else:
                    meaning = f"every {abs(spec)} steps"
        
                print(f"  {', '.join(codes)} [{meaning}]")

        if show_lists:
            print("Index lists:")
            for key, lst in self.index_lists.items():
                print(f"  {key}: {lst}")


    def count_action(self, code):
        """
        Return the total number of times `code` will be executed over all steps 0..total_steps-1.
        Runs in O(len(self.actions) + sum lengths of relevant index_lists).
        """
        count = 0
        for c, spec in self.actions:
            if c != code:
                continue

            if spec == 0:
                if self.total_steps > 0:
                    count += 1

            elif spec > 0:
                for idx in self.index_lists.get(spec, []):
                    if 0 <= idx < self.total_steps:
                        count += 1

            else:  # spec < 0  ->  period = -spec
                count += (self.total_steps - 1) // (-spec) + 1

        return count
