import { expect, test } from '@playwright/test'

test.describe('Painel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('mostra o titulo da home', async ({ page }) => {
    await expect(page.getByRole('heading')).toBeVisible()
  })
})
