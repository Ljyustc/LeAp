# -*- encoding:utf-8 -*-

def infix_to_postfix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return res

def postfix_to_prefix(post_equ, check=False):
    op_list = set(["+", "-", "*", "/", "^"])
    stack = []
    for elem in post_equ:
        sub_stack = []
        if elem not in op_list:
            sub_stack.append(elem)
            stack.append(sub_stack)
        else:
            if len(stack) >= 2:
                opnds = reversed([stack.pop() for i in range(2)])
                sub_stack.append(elem)
                for opnd in opnds:
                    sub_stack.extend(opnd)
                stack.append(sub_stack)
    if check and len(stack) != 1:
        pre_equ = None
    else:
        pre_equ = stack.pop()
    return pre_equ

def post_solver(post_equ):
    op_list = set(['+', '-', '/', '*', '^'])
    status = True
    stack = []
    for elem in post_equ:
        if elem in op_list:
            if len(stack) >= 2:
                op = elem
                opnd2 = stack.pop()
                opnd1 = stack.pop()
                if op == '+':
                    answer = opnd1 + opnd2
                elif op == '-':
                    answer = opnd1 - opnd2
                elif op == '*':
                    answer = opnd1 * opnd2
                elif op == '/':
                    answer = opnd1 / opnd2
                elif op == '^':
                    answer = opnd1 ** opnd2
                else:
                    status = False
                    break
                stack.append(answer)
            else:
                status = False
                break
        else:
            elem = float(elem)
            stack.append(elem)
    if status and len(stack) == 1:
        answer = stack.pop()
    else:
        answer = None
        status = False
    return status, answer

def number_map(equ, num_list):
    num_equ = []
    for token in equ:
        if "temp_" in token:
            token = num_list[ord(token[-1]) - ord('a')]
        elif token == "PI":
            token = 3.14
        num_equ.append(token)
    return num_equ

def eval_num_list(str_num_list):
    num_list = list()
    for item in str_num_list:
        if item[-1] == "%":
            num_list.append(float(item[:-1]) / 100)
        else:
            num_list.append(eval(item))
    return num_list
