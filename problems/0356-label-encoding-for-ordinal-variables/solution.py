def label_encode_ordinal(values: list, order: list) -> list:
    order_012 = []
    values_012 = []

    for index, name in enumerate(order):
        order_012.append((index, name))

    for i in range(len(values)):
        for j in range(len(order)):
            if values[i] == order_012[j][1]:
                values_012.append(order_012[j][0])
                break
        else:
            values_012.append(-1)

    return values_012