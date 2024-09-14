from .user.playground import UserPlayground
playground_list = {
    "playground": UserPlayground
}


def get_playground(playground_name):
    if playground_name in playground_list.keys():
        return playground_list[playground_name]
    else:
        raise Exception("Playground not found")
