# answer for the todo 'evaluate' - run this cell to define it
def evaluate(model, loader, limit=None):
    """accuracy on a dataset, plus a confusion matrix: confusion[true label][guessed label] = count"""
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    for k, (image, label) in enumerate(loader):
        if limit is not None and k >= limit:
            break
        guess, _ = model.predict(image)
        confusion[label][guess] += 1
    total = sum(map(sum, confusion))
    correct = sum(confusion[i][i] for i in range(len(CLASSES)))
    return correct / max(total, 1), confusion
