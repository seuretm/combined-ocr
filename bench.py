class Bench():
    def __init__(self, headers, sorting=min):
        self.headers = headers
        self.sorting = sorting
        self.content = None
        
    def __call__(self, value):
        if self.content is None:
            self.content = [[]]
        if len(self.content[-1])>=len(self.headers):
            self.content.append([])
        self.content[-1].append(value)

    def maximum(self, col=None):
        if col is None:
            return [self.maximum(i) for i in range(len(self.headers))]
        m = None
        for i in range(len(self.content)):
            if type(self.content[i][col])==str:
                continue
            if m is None:
                m = self.content[i][col]
            else:
                m = self.sorting(m, self.content[i][col])
        return m

    def mattermost(self, fmt='%.2f'):
        res = []
        res.append('|%s|' % ' | '.join(self.headers))
        res.append('|%s|' % ' | '.join([':---' for x in range(len(self.headers))]))
        m = self.maximum()
        for row in self.content:
            r = []
            for i, v in enumerate(row):
                if type(v)==str:
                    r.append(v)
                elif v==m[i]:
                    r.append((('**%s**')%fmt)%v)
                else:
                    r.append(fmt%v)
            res.append('|%s|' % ' | '.join(r))
        return '\n'.join(res)

if __name__=='__main__':
    bench = Bench(['method', 'dataset 1', 'dataset 2'])
    bench('hello')
    bench(12.3)
    bench(4.5)
    bench('world')
    bench(11.2)
    bench(5.2)
    print(bench.mattermost())

