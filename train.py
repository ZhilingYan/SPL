def train():
    model = Model(2, 2)
    dataloader = get_dataloader()
    criterion = SPLLoss(n_samples=len(dataloader.dataset))
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):
        for index, data, target in tqdm.tqdm(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target, index)
            loss.backward()
            optimizer.step()
        criterion.increase_threshold()
        plot(dataloader.dataset, model, criterion)

    animation = camera.animate()
    animation.save("plot.gif")
