import matplotlib.pyplot as plt

def plot_list(nums, title="Training Loss"):
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes()
    plt.title(title)
    #plt.setp(ax, xticks=(), yticks=())
    #plt.scatter(enumerate(nums))

    #plt.plot(range(len(nums)), nums, 'ro', markerfacecolor = 'none')
    plt.plot(range(len(nums)), nums)
    #ax.add_artist(imagebox)
    plt.savefig(title)
    plt.show()
    plt.close(fig)

if __name__ == '__main__':
    a = [1,2,5,2,10,3,8,5,3,4]
    plot_list(a)