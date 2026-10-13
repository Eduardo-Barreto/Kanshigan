import { expect, test } from '@playwright/test'

test.describe('Biblioteca', () => {
  test('mostra o gold em destaque com seus clips', async ({ page }) => {
    await page.goto('/')

    const gold = page.getByRole('region', { name: 'Gold', exact: true })
    await expect(gold.getByRole('link', { name: /gold_demo/ })).toBeVisible()
    await expect(gold.getByRole('meter', { name: 'Revisão de gold_demo' })).toHaveAttribute(
      'value',
      '100',
    )
  })

  test('separa os candidatos que esperam revisão', async ({ page }) => {
    await page.goto('/')

    const candidates = page.getByRole('region', { name: 'Candidatos a gold' })
    await expect(candidates.getByRole('link', { name: /cand_demo/ })).toBeVisible()
  })

  test('abre a bancada ao escolher um clip', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('link', { name: /gold_demo/ }).click()

    await expect(page.getByRole('img', { name: 'Quadro 1 de gold_demo' })).toBeVisible()
    await expect(page.getByRole('region', { name: 'Linha do tempo' })).toBeVisible()
  })
})
