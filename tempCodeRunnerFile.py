        for i in range(self.num_white_keys):
            if (i % 7) in [0, 1, 3, 4, 5]:  # C, D, F, G, A have black keys
                x_start = x0 + (i + 1) * white_key_width - black_key_width // 2
                y_start = y0
                x_end = x_start + black_key_width
                y_end = y0 + black_key_height
                black_key_polygon = [[x_start, y_start], [x_end, y_start], [x_end, y_end], [x_start, y_end]]
                self.black.append(np.array(black_key_polygon, dtype=np.int32))